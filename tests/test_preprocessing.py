import contextlib
import csv
import gzip
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from rescope.config import load_config
from rescope.dataset import OnlyVisiumHDRegionDataset
from rescope.geometry import tile_box_to_image_box, grid_tile_pixels
from rescope.preprocessing.pipeline import load_pipeline, make_rescope_config, run_pipeline, paths


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {k:v for k,v in os.environ.items() if k != 'ISCOUT_PROJECT_ROOT'}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()
        gc.collect()

    def pipeline_config(self):
        config = {'sample':'test', 'wsi_path':'image.tif', 'mpp':0.2738, 'tile_um':8,
                  'work_dir':'work', 'uni_checkpoint':'uni.bin', 'cellvit_repo':'CellViT',
                  'cellvit_checkpoint':'CellViT-SAM-H-x40.pth', 'cellvit_python':sys.executable,
                  'device':'cpu', 'reader':'tiff-memmap', 'rescope':{'model':{'base_roi_tiles':2,'base_stride_tiles':2},
                  'train':{'epochs':1}, 'selection':{'min_cells':1}}}
        path = self.root/'pipeline.yaml'
        path.write_text(yaml.safe_dump(config))
        return path

    def test_patch_coordinates_match_upstream_offsets(self):
        import tifffile
        from PIL import Image
        from rescope.preprocessing.patches import create_patches
        image = np.empty((1100,2100,3),dtype=np.uint8)
        image[...,0] = np.arange(2100,dtype=np.uint16)%230
        image[...,1] = (np.arange(1100,dtype=np.uint16)%230)[:,None]
        image[...,2] = 100
        wsi = self.root/'wide.tif'
        tifffile.imwrite(wsi,image,photometric='rgb')
        with contextlib.redirect_stdout(io.StringIO()):
            folder = create_patches(wsi,self.root/'patches',0.25,40,'tiff-memmap')
        meta=yaml.safe_load((folder/'metadata.yaml').read_text())
        self.assertEqual((meta['orig_n_tiles_rows'],meta['orig_n_tiles_cols']),(2,3))
        for row,col in ((0,1),(1,0),(1,2)):
            image_patch=np.asarray(Image.open(folder/'patches'/f'wide_{row}_{col}.png'))
            np.testing.assert_array_equal(image_patch[32,32],image[row*960,col*960])
            # CellViT's np.flip([row*960-32,col*960-32]) plus local xy.
            offset=np.array([col*960-32,row*960-32])
            np.testing.assert_array_equal(offset+[32,32],[col*960,row*960])
        with self.assertRaises(FileExistsError):
            create_patches(wsi,self.root/'patches',0.25,40,'tiff-memmap')

    def test_cell_table_keeps_xy_types_and_filters_padding(self):
        from rescope.preprocessing.cells import convert_cells
        source=self.root/'cells.json'
        source.write_text(json.dumps({'cells':[{'centroid':[12,3],'type':1},
            {'centroid':[6,9],'type':2},{'centroid':[-1,2],'type':3}]}))
        output=convert_cells(source,self.root/'out',20,10)
        with gzip.open(output,'rt') as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual([(float(r['centroid_x']),float(r['centroid_y']),int(r['cell_type'])) for r in rows],[(12,3,1),(6,9,2)])
        self.assertEqual(json.loads((output.parent/'summary.json').read_text())['outside_image'],1)

    def test_invalid_cell_type_never_publishes_table(self):
        from rescope.preprocessing.cells import convert_cells
        source=self.root/'cells.json'
        source.write_text(json.dumps({'cells':[{'centroid':[1,2],'type':1.5}]}))
        with self.assertRaises(ValueError):
            convert_cells(source,self.root/'out',20,10)
        self.assertFalse((self.root/'out/cell_metadata.csv.gz').exists())
        self.assertFalse((self.root/'out/cell_metadata.partial.csv.gz').exists())

    def test_hd_aggregation_retains_only_assigned_cells(self):
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        from scipy import sparse
        from rescope.preprocessing.cells import convert_cells
        square=self.root/'square_002um'
        (square/'spatial').mkdir(parents=True)
        rc=[(0,0),(0,1),(0,2),(1,0)]
        pq.write_table(pa.Table.from_pylist([dict(array_row=r,array_col=c,pxl_row_in_fullres=float(r),pxl_col_in_fullres=float(c)) for r,c in rc]),square/'spatial/tissue_positions.parquet')
        matrix=sparse.csc_matrix([[1,2,4,8],[10,20,40,80]],dtype=np.int32)
        with h5py.File(square/'filtered_feature_bc_matrix.h5','w') as h:
            g=h.create_group('matrix')
            for key,value in [('data',matrix.data),('indices',matrix.indices),('indptr',matrix.indptr),('shape',matrix.shape)]:g[key]=value
            g['barcodes']=np.array([f's_002um_{r:05d}_{c:05d}-1'.encode() for r,c in rc])
            feat=g.create_group('features')
            for key,value in [('id',[b'g1',b'g2']),('name',[b'A',b'B']),('feature_type',[b'Gene Expression']*2)]:feat[key]=value
        cells=[]
        for x,y in ((0,0),(2,0),(8,8)):
            cells.append({'centroid':[x,y],'contour':[[x-.2,y-.2],[x+.2,y-.2],[x+.2,y+.2],[x-.2,y+.2]],
                          'bbox':[[y-.2,x-.2],[y+.2,x+.2]],'type':1})
        source=self.root/'cells.json';source.write_text(json.dumps({'cells':cells}))
        with contextlib.redirect_stdout(io.StringIO()):
            output=convert_cells(source,self.root/'out',10,10,square)
        table=pd.read_csv(output)
        self.assertEqual(table.centroid_x.tolist(),[0,2])
        np.testing.assert_array_equal(sparse.load_npz(output.parent/'cell_gene_counts.npz').toarray(),[[1,10],[4,40]])
        self.assertAlmostEqual(table.bbox_xmin.iloc[1],1.8)

    def test_integer_tile_pitch_is_used_in_all_coordinate_conversions(self):
        cfg={'data':{'tile_um':8,'mpp':0.2738,'tile_px':29}}
        self.assertEqual(grid_tile_pixels(cfg),29)
        box=tile_box_to_image_box({'global_top_tile':100,'global_left_tile':200,'roi_tiles':164},8,0.2738,29)
        self.assertEqual((box['top_px'],box['left_px'],box['size_px']),(2900,5800,4756))
        self.assertAlmostEqual(grid_tile_pixels({'data':{'tile_um':8,'mpp':0.2738}}),8/0.2738)

    def test_full_grid_keeps_filtered_trailing_rows_and_columns(self):
        directory=self.root/'embeddings';directory.mkdir()
        np.save(directory/'embeddings.npy',np.zeros((2,8),dtype=np.float32))
        (directory/'meta.csv').write_text('row,col\n0,0\n1,2\n')
        ds=OnlyVisiumHDRegionDataset(str(directory),{'source':'ratio'},[5,7])
        self.assertEqual(ds.get_region_info()['full_grid_shape'],[5,7])
        with self.assertRaises(ValueError):
            OnlyVisiumHDRegionDataset(str(directory),{'source':'ratio'},[1,2])

    def test_config_generation_uses_actual_grid_and_rejects_other_slide(self):
        path=self.pipeline_config();cfg=load_pipeline(path);p=paths(cfg)
        p['embeddings'].mkdir(parents=True);p['cells'].mkdir()
        np.save(p['embeddings']/'embeddings.npy',np.zeros((2,1536),dtype=np.float32))
        manifest={'wsi_path':cfg['wsi_path'],'mpp':.2738,'tile_um':8,'tile_px':29,'num_tiles_y':5,'num_tiles_x':7,'tiles_embedded':2}
        (p['embeddings']/'manifest.json').write_text(json.dumps(manifest))
        with gzip.open(p['cells']/'cell_metadata.csv.gz','wt') as h:h.write('centroid_x,centroid_y,cell_type\n1,2,1\n4,5,2\n')
        output=make_rescope_config(cfg)
        result=load_config(output,num_rois=3)
        self.assertEqual(result['data']['tile_px'],29)
        self.assertEqual(result['data']['grid_shape'],[5,7])
        self.assertEqual(result['data']['embedding_dir'],str(p['embeddings']))
        output.unlink();manifest['wsi_path']=str(self.root/'other.tif')
        (p['embeddings']/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'different WSI'):make_rescope_config(cfg)

    def test_dry_run_is_read_only_and_shows_both_environments(self):
        path=self.pipeline_config()
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):run_pipeline(path,dry_run=True)
        self.assertIn('rescope.preprocessing.uni',stream.getvalue())
        self.assertIn('cellvit.py',stream.getvalue())
        self.assertFalse((self.root/'work').exists())

    def test_generated_inputs_feed_all_three_independent_models(self):
        from rescope.preprocessing.cells import convert_cells
        from rescope.batch import run_budgets
        path=self.pipeline_config()
        raw=yaml.safe_load(path.read_text())
        raw["rescope"]["model"]["prototype_grid_size"]=2
        raw["rescope"]["selection"].update(projection_dim=4,n_clusters=4,candidate_pool_size=16,
            target_reference_cells=20,quota_reference_cells=20,min_reference_fraction=0,
            enforce_celltype_upper_bounds=False)
        raw["rescope"]["runtime"]={"threads":1}
        path.write_text(yaml.safe_dump(raw))
        cfg=load_pipeline(path);p=paths(cfg);p["embeddings"].mkdir(parents=True)
        rng=np.random.default_rng(42)
        np.save(p["embeddings"]/"embeddings.npy",rng.normal(size=(64,1536)).astype(np.float32))
        (p["embeddings"]/"meta.csv").write_text("row,col\n"+"".join(f"{r},{c}\n" for r in range(8) for c in range(8)))
        (p["embeddings"]/"manifest.json").write_text(json.dumps({"wsi_path":cfg["wsi_path"],"mpp":.2738,
            "tile_um":8,"tile_px":29,"num_tiles_y":8,"num_tiles_x":8,"tiles_embedded":64}))
        source=self.root/"cells.json"
        source.write_text(json.dumps({"cells":[{"centroid":[c*29+10,r*29+10],"type":1+(r+c)%2} for r in range(8) for c in range(8)]}))
        convert_cells(source,p["cells"],232,232)
        output=make_rescope_config(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            result=run_budgets(output,output_dir=p["results"])
        for k in (1,2,3):
            metadata=json.loads((result/f"roi_{k}"/"visium_hd_region_eval_metadata.json").read_text())
            self.assertEqual(len(metadata["model_rois"]),k)
            self.assertEqual(metadata["checkpoint_provenance"]["train_num_rois"],k)
            for box in metadata["model_rois"]:
                self.assertEqual(box["size_px"],58)
                self.assertEqual(box["left_px"]%29,0)
                self.assertEqual(box["top_px"]%29,0)

    def test_uni_writer_keeps_row_order_and_manifest(self):
        import tifffile
        from rescope.preprocessing import uni
        image=np.full((18,30,3),100,dtype=np.uint8)
        wsi=self.root/'small.tif';tifffile.imwrite(wsi,image,photometric='rgb')
        weight=self.root/'stub.bin';weight.touch()
        output=self.root/'embeddings'
        # A small deterministic encoder tests the actual I/O loop without
        # pretending to validate the gated UNI2-h weights or biological quality.
        class Encoder(torch.nn.Module):
            def forward(self,x):return x.mean((2,3))
        argv=['uni','--wsi-path',str(wsi),'--sample','small','--output-dir',str(output),'--weight-path',str(weight),
              '--tile-um','3','--mpp','1','--batch-size','8','--num-workers','0','--reader','tiff-memmap','--device','cpu']
        with patch.object(sys,'argv',argv),patch.object(uni,'build_uni_model',return_value=Encoder()),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            uni.main()
        matrix=np.load(output/'embeddings.npy')
        self.assertEqual(matrix.shape,(60,3))
        with (output/'meta.csv').open() as handle:rows=list(csv.DictReader(handle))
        self.assertEqual([(int(r['row']),int(r['col'])) for r in rows],[(r,c) for r in range(6) for c in range(10)])
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual((manifest['tile_px'],manifest['num_tiles_y'],manifest['num_tiles_x']),(3,6,10))
        self.assertTrue(np.isfinite(matrix).all())
