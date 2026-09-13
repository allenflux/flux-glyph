"""Sealed Android evaluation integrity; synthetic fixtures and mocked ONNX only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from training import evaluate_android_regions as evaluate
from training.capture import capture_android


FAMILIES=[f'Font {index}' for index in range(9)]+[evaluate.UNKNOWN]
GATES={'min_score':.5,'min_margin':.01,'min_patch_agreement':2/3}


class AndroidEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.args=SimpleNamespace(**{k:self.root/k for k in ('run','data','region')},output=self.root/'TEST.json')
        for key in ('run','data','region'):getattr(self.args,key).mkdir()
        (self.args.run/'model.pth').write_bytes(b'mocked frozen checkpoint')
        (self.args.region/'model.onnx').write_bytes(b'mocked frozen ONNX')
        evaluate.dump(self.args.data/'MANIFEST.json',{'fixture':True})
        self.selection={'families':FAMILIES,'data_manifest_sha256':evaluate.sha(self.args.data/'MANIFEST.json'),
                        'selected':{'temperature':1.,'gates':GATES}}
        evaluate.dump(self.args.run/'SELECTION.json',self.selection)
        self.meta={'temperature':1.,'gates':GATES,'max_size_relative_spread':evaluate.POLICY['max_size_relative_spread'],
            'training':{'selection_sha256':evaluate.sha(self.args.run/'SELECTION.json'),
                        'checkpoint_sha256':evaluate.sha(self.args.run/'model.pth'),
                        'data_manifest_sha256':self.selection['data_manifest_sha256']}}
        evaluate.dump(self.args.region/'metadata.json',self.meta)
        self.parity={'schema':'flux-glyph-android-onnx-parity-v1','passed':True,'test_read':False,
            'selection_sha256':evaluate.sha(self.args.run/'SELECTION.json'),
            'checkpoint_sha256':evaluate.sha(self.args.run/'model.pth'),
            'model_sha256':evaluate.sha(self.args.region/'model.onnx'),
            'metadata_sha256':evaluate.sha(self.args.region/'metadata.json'),
            'source_bindings':{str(p.resolve()):evaluate.sha(p) for p in (evaluate.ROOT/'training/export_android_regions.py',
                evaluate.ROOT/'training/export_region_stable.py',evaluate.ROOT/'training/train_regions.py')},
            'calibration_font_decisions_identical':True,'original_torch_groupnorm_reference':True,
            'calibration_size_availability_identical':True,'calibration_size_values_close':True,'size_value_atol_px':.02,
            'size_value_rtol':.0003,'cached_mps_outputs_reproduce_selected_metrics':True,
            'batch_checks':[{'batch_size':n,'samples':40,'passed':True,'font_and_size_checked':True} for n in (1,7,32,128)]}
        self.write_parity()
        source_rows=[];rows=[]
        for split in ('train','calibration','test'):
            for target,family in enumerate(FAMILIES):
                actual=family if family!=evaluate.UNKNOWN else 'Unknown '+split
                native={'training_family':family,'font_family':actual,'font_file_sha256':hashlib.sha256(actual.encode()).hexdigest(),
                        'split':split,'source_id':split+'-page','region_id':str(target)}
                source_rows.append(native)
                if split=='test':
                    for view in evaluate.RECIPES:
                        rows.append({'family':family,'target':target,'view':view,'source_id':native['source_id'],
                            'region_id':native['region_id'],'source_font_family':actual,'font_file_sha256':native['font_file_sha256'],
                            'tile_start':len(rows),'tile_count':1,'ink_height_px':40,'font_size_px':40})
        tiles=np.zeros((40,1,64,256),dtype=np.float32);tiles[:,0,0,0]=np.arange(40)
        folder=self.args.data/'test';folder.mkdir();(folder/'tiles.raw').write_bytes(tiles.tobytes());evaluate.dump(folder/'rows.json',rows)
        partition={'frozen_selection':{'sha256':evaluate.sha(self.args.run/'SELECTION.json')},'rejected':[],
                   'array':{'path':'tiles.raw','sha256':evaluate.sha(folder/'tiles.raw')},
                   'metadata':{'path':'rows.json','sha256':evaluate.sha(folder/'rows.json')}}
        evaluate.dump(folder/'MANIFEST.json',partition)
        self.data={'families':FAMILIES,'manifest_sha256':self.selection['data_manifest_sha256'],'partition':partition,
            'partition_sha256':evaluate.sha(folder/'MANIFEST.json'),'rows':rows,'tiles':tiles,
            'manifest':{'views':evaluate.RECIPES,'capture':str(self.root/'capture'),'capture_manifest_sha256':'c'*64}}
        self.source={'manifest_sha256':'c'*64,'rows':source_rows}
        self.model=SimpleNamespace(output_families=FAMILIES,meta=self.meta,session=SimpleNamespace(run=self.infer))
        self.validator=patch.object(evaluate,'validate',side_effect=lambda run,data:deepcopy(self.selection)).start()
        self.loader=patch.object(evaluate,'load_split',side_effect=self.load_test).start()
        self.model_loader=patch('flux_glyph.android_font.AndroidFontClassifier',return_value=self.model).start()
        patch.object(capture_android,'load_capture',side_effect=lambda root:deepcopy(self.source)).start()

    def tearDown(self):
        patch.stopall();self.temp.cleanup()

    def write_parity(self):evaluate.dump(self.args.run/'PARITY.json',self.parity)

    def load_test(self,root,split):
        self.assertEqual(split,'test');self.assertTrue((self.root/'TEST_FREEZE.json').is_file())
        return self.data

    def infer(self,names,inputs):
        self.assertEqual(names,['logits','log_em_ratio']);indices=inputs['tiles'][:,0,0,0].astype(int)
        logits=np.full((len(indices),10),-10,dtype=np.float32)
        for index,source_index in enumerate(indices):logits[index,self.data['rows'][source_index]['target']]=10.
        return [logits,np.zeros(len(indices),dtype=np.float32)]

    def test_complete_eval_is_sealed_and_by_view_is_diagnostic_only(self):
        result=evaluate.evaluate(self.args)
        self.assertTrue(result['passed']);self.assertEqual(result['metrics']['views'],40)
        self.assertEqual(result['denominators']['unknown_views_including_rejected'],4)
        self.assertTrue(result['holdout_evidence']['whole_unknown_families_and_files_disjoint'])
        self.assertFalse(result['denominators']['color_accuracy_evaluated'])
        for view,value in result['by_view'].items():
            self.assertIsNone(value['passed']);self.assertTrue(value['diagnostic_only'])
            if view!='native':
                self.assertIsNone(value['native_known_top1_accuracy']);self.assertIsNone(value['checks']['native_known_top1'])
        with self.assertRaisesRegex(ValueError,'existing sealed test report'):evaluate.evaluate(self.args)
        self.assertEqual(self.loader.call_count,1)

    def test_missing_font_size_or_batch_parity_blocks_before_model_and_test(self):
        original=deepcopy(self.parity)
        for change in ({'calibration_size_values_close':False},{'calibration_size_availability_identical':False},
                       {'calibration_font_decisions_identical':False},{'size_value_atol_px':1.},
                       {'cached_mps_outputs_reproduce_selected_metrics':False},{'size_value_rtol':.1},
                       {'batch_checks':[]},{'test_read':True},{'source_bindings':{}}):
            self.parity={**deepcopy(original),**change};self.write_parity()
            with self.subTest(change=change),self.assertRaises(ValueError):evaluate.evaluate(self.args)
        self.loader.assert_not_called();self.model_loader.assert_not_called()

    def test_checkpoint_or_export_model_tamper_cannot_open_test(self):
        for path in (self.args.run/'model.pth',self.args.region/'model.onnx',self.args.region/'metadata.json'):
            before=path.read_bytes();path.write_bytes(before+b'tampered')
            with self.subTest(path=path),self.assertRaises(ValueError):evaluate.evaluate(self.args)
            path.write_bytes(before)
        self.loader.assert_not_called()

    def test_changed_runtime_gate_or_provenance_cannot_open_test(self):
        original=deepcopy(self.meta)
        for change in ({'temperature':2.},{'gates':{**GATES,'min_score':.1}},{'training':{}}):
            self.model.meta={**deepcopy(original),**change}
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'runtime|family order'):
                evaluate.evaluate(self.args)
        self.loader.assert_not_called()

    def test_parity_checkpoint_selection_or_freeze_mutation_during_inference_never_publishes(self):
        paths=(self.args.run/'PARITY.json',self.args.run/'model.pth',self.args.run/'SELECTION.json',self.root/'TEST_FREEZE.json')
        for path in paths:
            previous=path.read_bytes() if path.exists() else None
            def infer(names,inputs):
                result=self.infer(names,inputs);path.write_bytes(path.read_bytes()+b'\n');return result
            self.model.session.run=infer
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'source/model changed during TEST'):
                evaluate.evaluate(self.args)
            self.assertFalse(self.args.output.exists())
            if previous is not None:path.write_bytes(previous)
            (self.root/'TEST_FREEZE.json').unlink()

    def test_failed_inference_leaves_a_freeze_that_blocks_silent_rerun(self):
        def fail(names,inputs):raise RuntimeError('mocked runtime failure')
        self.model.session.run=fail
        with self.assertRaisesRegex(RuntimeError,'mocked runtime failure'):evaluate.evaluate(self.args)
        self.model.session.run=self.infer
        with self.assertRaisesRegex(ValueError,'already started'):evaluate.evaluate(self.args)
        self.assertEqual(self.loader.call_count,1);self.assertFalse(self.args.output.exists())

    def test_dataset_bytes_changed_during_inference_never_publish(self):
        def infer(names,inputs):
            result=self.infer(names,inputs)
            with (self.args.data/'test/tiles.raw').open('ab') as stream:stream.write(b'changed')
            return result
        self.model.session.run=infer
        with self.assertRaisesRegex(ValueError,'TEST data changed'):evaluate.evaluate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_unknown_holdout_checks_whole_family_and_file_independence(self):
        original=deepcopy(self.source)
        for field in ('font_family','font_file_sha256'):
            self.source=deepcopy(original);self.source['rows'][29][field]=self.source['rows'][9][field]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'holdout crosses partitions'):
                evaluate.holdout_evidence(self.data)

    def test_unknown_population_cannot_be_silently_dropped_from_test(self):
        self.source['rows'].append({**self.source['rows'][-1],'region_id':'unaccounted-unknown'})
        with self.assertRaisesRegex(ValueError,'unknown population differs'):evaluate.holdout_evidence(self.data)

    def test_rejections_are_in_confusion_coverage_and_unknown_outcomes(self):
        rows=[self.data['rows'][0],self.data['rows'][36]]
        predictions=[{'predicted':0,'score':.99,'margin':.98,'agreement':1.,'em_ratio':1.,'size_spread':0.},
                     {'predicted':1,'score':.4,'margin':.1,'agreement':1.,'em_ratio':1.,'size_spread':0.}]
        details=evaluate.decisions(predictions,rows,FAMILIES,GATES)
        rejected=[{'family':FAMILIES[0],'view':'native'},{'family':evaluate.UNKNOWN,'view':'native'}]
        measured,views,confusion,denom=evaluate.summarize(details,FAMILIES,rejected,evaluate.RECIPES)
        self.assertEqual(measured['known_views'],2);self.assertEqual(measured['known_correct_coverage'],.5)
        self.assertEqual(measured['native_known_top1_accuracy'],.5)
        self.assertEqual(measured['unknown_not_named_rate'],1.);self.assertEqual(measured['explicit_unknown_recall'],0.)
        self.assertEqual(sum(sum(row.values()) for row in confusion.values()),4)
        self.assertEqual(denom['unknown_outcomes'],{'wrongly_named':0,'explicit_unknown':0,'uncertain_without_name':1,'preprocessing_rejected':1})
        self.assertEqual(denom['size_correct_coverage_of_all_known_views'],.5)

    def test_sizes_on_wrong_names_do_not_improve_correct_size_metrics(self):
        rows=[self.data['rows'][0],self.data['rows'][4],self.data['rows'][36]]
        predictions=[{'predicted':1,'score':.99,'margin':.98,'agreement':1.,'em_ratio':1.,'size_spread':0.},
                     {'predicted':1,'score':.99,'margin':.98,'agreement':1.,'em_ratio':1.,'size_spread':.21},
                     {'predicted':9,'score':.99,'margin':.98,'agreement':1.,'em_ratio':1.,'size_spread':0.}]
        details=evaluate.decisions(predictions,rows,FAMILIES,GATES)
        measured,_,_,denom=evaluate.summarize(details,FAMILIES,[],evaluate.RECIPES)
        self.assertEqual(measured['size']['correctly_named_with_size'],0)
        self.assertIsNone(measured['size']['median_ape']);self.assertEqual(denom['size_available_on_wrong_names'],1)


if __name__=='__main__':unittest.main()
