"""Fail-closed audits for new Android data; all native collection is mocked.

These fixtures are small artificial shapes, never phone screenshots or held-out
model data. Real preprocessing is exercised after mocked native verification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from training import prepare_android_regions as data
from training.capture import capture_android as capture


FAMILIES=[f'Known {index}' for index in range(9)]+[data.UNKNOWN]


class AndroidDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.capture=self.root/'capture';self.capture.mkdir();self.output=self.root/'data'
        self.rows=[];self.verified=[]
        for index,split in enumerate(data.SPLITS):
            image=Image.new('RGB',(131,47),'white');draw=ImageDraw.Draw(image)
            for number in range(6):
                x=8+number*18
                draw.rectangle((x,8+index,x+4,37),fill='#333333')
                draw.rectangle((x,10+number,x+10,14+number+index),fill='#333333')
            path=self.capture/(split+'.png');image.save(path)
            for target,family in enumerate(FAMILIES):
                actual=family if family!=data.UNKNOWN else 'Unknown '+split
                self.rows.append({'split':split,'training_family':family,'font_family':actual,
                    'source_id':split,'region_id':f'{split}-r{target}','page_id':split+'-page',
                    'content_group_id':split+'-content','image':path.name,'image_sha256':data.sha(path),
                    'image_width':131,'image_height':47,'bbox':[0,0,131,47],'ink_bbox':[8,7.5+index,109,38],
                    'font_id':actual+'-regular','font_face':actual+'-Regular',
                    'font_file_sha256':hashlib.sha256(actual.encode()).hexdigest(),'ttc_index':0,
                    'font_size_screen_px':36.,'color':'#333333','text':split+' caption',
                    'native_font_verified':True})
        data.dump(self.capture/'CAPTURE_MANIFEST.json',{'fixture':True})
        self.load=patch.object(capture,'load_capture',side_effect=self.load_capture).start()
        self.verify=patch.object(capture,'verify_region',side_effect=self.verify_region).start()

    def tearDown(self):
        patch.stopall();self.temp.cleanup()

    def load_capture(self,root):
        self.assertEqual(Path(root),self.capture)
        return {'families':FAMILIES,'rows':deepcopy(self.rows),'bindings':{},
                'manifest_sha256':data.sha(self.capture/'CAPTURE_MANIFEST.json')}

    def verify_region(self,root,row):
        self.assertEqual(Path(root),self.capture);self.verified.append(row['split'])

    def prepare(self,split='train',selection=None):
        return data.prepare(self.capture,self.output,split,selection)

    def audit(self,rows=None):
        return data.audit_identities(self.rows if rows is None else rows,FAMILIES)

    def selection(self):
        path=self.root/'selection.json'
        data.dump(path,{'schema':'flux-glyph-android-training-selection-v1','passed':True,
                       'test_read':False,'data_manifest_sha256':data.sha(self.output/'MANIFEST.json')})
        return path

    def rewrite_rows(self,change,split='train'):
        folder=self.output/split;rows=json.loads((folder/'rows.json').read_text());change(rows)
        data.dump(folder/'rows.json',rows);manifest=json.loads((folder/'MANIFEST.json').read_text())
        manifest['metadata']['sha256']=data.sha(folder/'rows.json');data.dump(folder/'MANIFEST.json',manifest)

    def test_valid_sources_share_known_fonts_but_keep_unknown_sources_disjoint(self):
        self.assertEqual(sum(self.audit().values()),30)

    def test_source_and_content_identities_cannot_cross_splits(self):
        for key in ('source_id','page_id','content_group_id','image','image_sha256'):
            rows=deepcopy(self.rows);rows[10][key]=rows[0][key]
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'crosses partitions|inconsistent native'):
                self.audit(rows)

    def test_text_identity_normalizes_case_width_and_whitespace(self):
        rows=deepcopy(self.rows);rows[0]['text']=' ＡＣＣＯＵＮＴ\t Total '
        rows[10]['text']='accounttotal'
        with self.assertRaisesRegex(ValueError,'crosses partitions: text'):self.audit(rows)

    def test_whole_unknown_family_holdout_includes_different_files_and_weights(self):
        rows=deepcopy(self.rows);rows[19]['font_family']=rows[9]['font_family']
        self.assertNotEqual(rows[19]['font_file_sha256'],rows[9]['font_file_sha256'])
        with self.assertRaisesRegex(ValueError,'unknown font source/family crosses'):self.audit(rows)

    def test_unknown_ttc_file_cannot_cross_splits_via_another_index(self):
        rows=deepcopy(self.rows);rows[19]['font_file_sha256']=rows[9]['font_file_sha256'];rows[19]['ttc_index']=1
        with self.assertRaisesRegex(ValueError,'unknown font source/family crosses'):self.audit(rows)

    def test_unknown_file_cannot_reappear_as_known_different_ttc_face(self):
        rows=deepcopy(self.rows);rows[10]['font_file_sha256']=rows[9]['font_file_sha256'];rows[10]['ttc_index']=1
        with self.assertRaisesRegex(ValueError,'unknown font source file appears'):self.audit(rows)

    def test_same_native_font_face_cannot_be_assigned_two_training_classes(self):
        rows=deepcopy(self.rows);rows[1]['font_file_sha256']=rows[0]['font_file_sha256']
        with self.assertRaisesRegex(ValueError,'font face has conflicting training classes'):self.audit(rows)

    def test_same_family_cannot_be_known_and_unknown_under_a_different_file(self):
        rows=deepcopy(self.rows);rows[9]['font_family']=rows[0]['font_family']
        with self.assertRaisesRegex(ValueError,'font family has conflicting training classes'):self.audit(rows)

    def test_native_geometry_style_and_identity_fail_before_any_pixels_are_opened(self):
        changes=[{'bbox':[-1,0,131,47]},{'bbox':[0,0,0,47]},{'bbox':[0,0,132,47]},
                 {'bbox':[0,0,131.,47]},{'bbox':[False,0,131,47]},{'image_width':True},
                 {'font_size_screen_px':float('nan')},{'font_size_screen_px':float('inf')},
                 {'font_size_screen_px':0},{'font_size_screen_px':True},{'font_size_screen_px':513},
                 {'color':'red'},{'font_file_sha256':'wrong'},{'ttc_index':-1},{'native_font_verified':False},
                 {'ink_bbox':[0,0,0,1]},{'ink_bbox':[-1,0,100,38]},{'ink_bbox':[8,8,132,38]},
                 {'ink_bbox':[float('nan'),8,109,38]},{'ink_bbox':[8,True,109,38]},{'ink_bbox':None},
                 {'image':'../outside.png'}]
        original=deepcopy(self.rows)
        for change in changes:
            self.rows=deepcopy(original);self.rows[20].update(change)
            with self.subTest(change=change),patch.object(data.Image,'open') as opened:
                with self.assertRaises(ValueError):self.prepare()
                opened.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_same_image_path_cannot_have_conflicting_hash_or_dimensions(self):
        for change in ({'image_sha256':'0'*64},{'image_width':132}):
            rows=deepcopy(self.rows);rows[1].update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'inconsistent native image identity'):
                self.audit(rows)

    def test_test_guard_precedes_capture_metadata_and_pixels(self):
        with self.assertRaisesRegex(ValueError,'requires a frozen passed CAL selection'):self.prepare('test')
        self.load.assert_not_called();self.verify.assert_not_called();self.assertFalse(self.output.exists())

    def test_failed_or_test_selected_selection_cannot_open_test(self):
        path=self.root/'selection.json'
        for content in ({'passed':False,'test_read':False},{'passed':True,'test_read':True},{}):
            data.dump(path,{'schema':'flux-glyph-android-training-selection-v1',**content})
            with self.subTest(content=content),self.assertRaisesRegex(ValueError,'frozen successful calibration'):
                self.prepare('test',path)
        self.load.assert_not_called();self.verify.assert_not_called()

    def test_roundtrip_binds_native_truth_four_views_and_actual_resize(self):
        result=self.prepare();before=data.sha(self.output/'MANIFEST.json');train=data.sha(self.output/'train/MANIFEST.json')
        self.prepare('calibration');loaded=data.load_split(self.output,'train')
        self.assertEqual(before,data.sha(self.output/'MANIFEST.json'))
        self.assertEqual(train,data.sha(self.output/'train/MANIFEST.json'))
        self.assertEqual(result['native_regions'],10);self.assertEqual(len(loaded['rows']),40)
        self.assertIsInstance(loaded['tiles'],np.memmap);self.assertEqual(loaded['families'],FAMILIES)
        half=next(row for row in loaded['rows'] if row['view']=='half')
        self.assertEqual(half['source_crop_bbox'],[2,1,115,44])
        self.assertAlmostEqual(half['font_size_px'],36*22/43)
        self.assertAlmostEqual(np.exp(half['log_em_ratio'])*half['ink_height_px'],half['font_size_px'])
        self.assertNotIn('test',self.verified)

    def test_native_ink_margin_rounds_outward_and_clamps_to_layout(self):
        row={'bbox':[10,20,100,80],'ink_bbox':[10.1,20.2,99.8,79.9],'font_size_screen_px':4}
        self.assertEqual(data.source_crop_bbox(row),row['bbox'])
        row.update(ink_bbox=[40.5,30.5,60.2,50.2],font_size_screen_px=16)
        self.assertEqual(data.source_crop_bbox(row),[37,27,64,54])
        row['font_size_screen_px']=84
        self.assertEqual(data.source_crop_bbox(row),[27,20,74,64])

    def test_crop_box_or_pixel_hash_tamper_fails_even_after_metadata_rehash(self):
        self.prepare();original=json.loads((self.output/'train/rows.json').read_text())
        for change in ({'source_crop_bbox':[0,0,131,47]},{'native_crop_sha256':'0'*64}):
            self.rewrite_rows(lambda rows:rows.__setitem__(slice(None),deepcopy(original)))
            self.rewrite_rows(lambda rows:rows[0].update(change))
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'crop differs from native ink bounds/pixels'):
                data.load_split(self.output,'train')

    def test_old_layout_scale_target_cannot_replace_tight_crop_scale(self):
        self.prepare()
        def tamper(rows):
            half=next(r for r in rows if r['view']=='half')
            half['font_size_px']=36*24/47
            half['log_em_ratio']=float(np.log(half['font_size_px']/half['ink_height_px']))
        self.rewrite_rows(tamper)
        with self.assertRaisesRegex(ValueError,'native size target differs'):data.load_split(self.output,'train')

    def test_native_font_verification_failure_cannot_produce_partition(self):
        self.verify.side_effect=ValueError('font fallback detected')
        with self.assertRaisesRegex(ValueError,'font fallback'):self.prepare()
        self.assertFalse((self.output/'train').exists())

    def test_source_image_bytes_tamper_is_rejected_during_prepare_and_load(self):
        self.prepare()
        with (self.capture/'train.png').open('ab') as stream:stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'native screenshot changed'):data.load_split(self.output,'train')
        with self.assertRaisesRegex(ValueError,'native screenshot changed'):
            data.prepare(self.capture,self.root/'other','train')

    def test_array_bytes_tamper_fails_even_if_outer_file_hash_is_updated(self):
        self.prepare();folder=self.output/'train';path=folder/'tiles.raw'
        array=np.memmap(path,mode='r+',dtype='<f4');array[0]=.123;array.flush();del array
        with self.assertRaisesRegex(ValueError,'prepared partition bytes changed'):data.load_split(self.output,'train')
        manifest=json.loads((folder/'MANIFEST.json').read_text());manifest['array']['sha256']=data.sha(path)
        data.dump(folder/'MANIFEST.json',manifest)
        with self.assertRaisesRegex(ValueError,'tensor pixels/hash differ'):data.load_split(self.output,'train')

    def test_relabel_with_matching_target_and_file_hash_still_fails_native_truth(self):
        self.prepare();self.rewrite_rows(lambda rows:rows[0].update(family=FAMILIES[1],target=1))
        with self.assertRaisesRegex(ValueError,'native truth/source differs'):data.load_split(self.output,'train')

    def test_size_color_and_ttc_metadata_tamper_cannot_override_native_truth(self):
        self.prepare();path=self.output/'train/rows.json';original=json.loads(path.read_text())
        for change in ({'font_size_px':25.},{'text_color_hex':'#000000'},{'ttc_index':1}):
            self.rewrite_rows(lambda rows:rows.__setitem__(slice(None),deepcopy(original)))
            self.rewrite_rows(lambda rows:rows[0].update(change))
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'native truth/source differs|native size target differs'):
                data.load_split(self.output,'train')

    def test_test_append_binds_selection_and_rechecks_it_before_load(self):
        self.prepare();selection=self.selection();self.prepare('test',selection)
        self.assertEqual(data.load_split(self.output,'test')['partition']['frozen_selection']['sha256'],data.sha(selection))
        self.verified.clear();selection.write_text(selection.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'frozen calibration selection changed'):data.load_split(self.output,'test')
        self.assertFalse(self.verified)

    def test_test_partition_cannot_drop_the_selection_hash(self):
        self.prepare();selection=self.selection();self.prepare('test',selection)
        path=self.output/'test/MANIFEST.json';manifest=json.loads(path.read_text())
        manifest['frozen_selection'].pop('sha256');data.dump(path,manifest);self.verified.clear()
        with self.assertRaisesRegex(ValueError,'lacks a frozen calibration selection'):data.load_split(self.output,'test')
        self.assertFalse(self.verified)

    def test_wrong_data_selection_cannot_verify_or_decode_test(self):
        self.prepare();selection=self.selection();value=json.loads(selection.read_text());value['data_manifest_sha256']='0'*64
        data.dump(selection,value);self.verified.clear()
        with self.assertRaisesRegex(ValueError,'TEST source differs'):self.prepare('test',selection)
        self.assertFalse(self.verified);self.assertFalse((self.output/'test').exists())

    def test_selection_mutation_during_test_preparation_does_not_commit(self):
        self.prepare();selection=self.selection()
        def mutate(root,row):
            if row['split']=='test':selection.write_text(selection.read_text()+'\n')
        self.verify.side_effect=mutate
        with self.assertRaisesRegex(ValueError,'frozen calibration selection changed'):self.prepare('test',selection)
        self.assertFalse((self.output/'test').exists())

    def test_existing_partition_cannot_be_overwritten(self):
        self.prepare();before=data.sha(self.output/'train/tiles.raw')
        with self.assertRaisesRegex(ValueError,'partition must be new'):self.prepare()
        self.assertEqual(before,data.sha(self.output/'train/tiles.raw'))

    def test_identical_crop_pixels_across_different_encoded_sources_are_rejected(self):
        self.prepare()
        with Image.open(self.capture/'train.png') as image:image.save(self.capture/'calibration.png',compress_level=0)
        digest=data.sha(self.capture/'calibration.png')
        for row in self.rows:
            if row['split']=='calibration':
                row['image_sha256']=digest;row['ink_bbox']=deepcopy(self.rows[0]['ink_bbox'])
        with self.assertRaisesRegex(ValueError,'identical crop pixels cross partitions'):self.prepare('calibration')

    def test_bound_preparation_source_tamper_is_rejected(self):
        self.prepare();(self.capture/'CAPTURE_MANIFEST.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'bound preparation source changed'):data.load_split(self.output,'train')


if __name__=='__main__':unittest.main()
