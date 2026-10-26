"""Guard the real saved-model demo's new interactions; never train a model."""
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest
from demo import app
from demo.presentation import EXAMPLES, SCENES, prediction_key, replay_table

ROOT = Path(__file__).resolve().parents[1]


class PredictionExperienceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = app.load_bundle(ROOT / 'demo/bundle')

    def test_predictions_still_match_all_reviewed_examples_without_fitting(self):
        batch = app._sample_batch(self.bundle)
        with patch.object(RandomForestClassifier, 'fit', side_effect=AssertionError('No training in the demo')):
            output = app.predict(batch, self.bundle, 'CT1')
        expected = self.bundle.sample_predictions.query("model == 'CT1'")
        self.assertEqual(output.predicted_label.tolist(), expected.predicted_label.tolist())
        changed = app.InputBatch(batch.baseline, batch.sample_ids,
                                  pd.Series(['BENIGN'] * len(batch.sample_ids)))
        self.assertEqual(app.predict(changed, self.bundle, 'CT1').predicted_label.tolist(),
                         output.predicted_label.tolist())

    def test_result_identity_includes_inputs_model_ids_and_comparison_labels(self):
        batch = app._subset_batch(self.bundle, [next(iter(EXAMPLES))])
        key = prediction_key(batch.baseline, batch.sample_ids, 'model-one', batch.actual_labels)
        different = batch.baseline.copy()
        different.loc[0, 'duration_seconds'] += 1
        for candidate in (
            prediction_key(different, batch.sample_ids, 'model-one', batch.actual_labels),
            prediction_key(batch.baseline, ['different-id'], 'model-one', batch.actual_labels),
            prediction_key(batch.baseline, batch.sample_ids, 'model-two', batch.actual_labels),
            prediction_key(batch.baseline, batch.sample_ids, 'model-one', ['Bot']),
        ):
            self.assertNotEqual(key, candidate)

    def test_replay_uses_real_predictions_with_unique_fictional_event_ids(self):
        source = app.predict(app._sample_batch(self.bundle), self.bundle, 'CT1')
        replay = replay_table(source)
        self.assertEqual(len(replay), 12)
        self.assertTrue(replay.event_id.is_unique)
        self.assertTrue(replay.sample_id.duplicated().any())  # explicitly disclosed replay
        by_id = source.set_index('sample_id')
        for row in replay.itertuples():
            self.assertEqual(row.predicted_label, by_id.loc[row.sample_id, 'predicted_label'])
            self.assertEqual(row.actual_label, by_id.loc[row.sample_id, 'actual_label'])
            self.assertEqual(row.needs_review, row.predicted_label != 'BENIGN')
        self.assertTrue((replay.actual_label.eq('BENIGN') & replay.needs_review).any())
        self.assertTrue((replay.actual_label.ne('BENIGN') & ~replay.needs_review).any())
        with self.assertRaises(ValueError):
            replay_table(source.iloc[0:1])

    def test_prediction_flow_refreshes_on_new_example_and_persists_on_rerun(self):
        ui = AppTest.from_file(str(ROOT / 'demo/app.py'), default_timeout=40).run()
        self.assertFalse(ui.exception)
        self.assertEqual([b.label for b in ui.button], ['Predict traffic'])
        self.assertEqual(len(ui.metric), 0)  # research grid no longer blocks the primary action
        ui.button(key='predict_traffic').click().run()
        self.assertFalse(ui.exception)
        first = ui.session_state['prediction_result']['key']
        ui.run()
        self.assertEqual(ui.session_state['prediction_result']['key'], first)
        ui.selectbox(key='example_id').select('validation-068045').run()
        # A result from the previous example is not displayed beside the new input.
        html = ' '.join(str(x.value) for x in ui.get('html'))
        self.assertIn('Ready when you are', html)
        ui.button(key='predict_traffic').click().run()
        self.assertFalse(ui.exception)
        self.assertNotEqual(ui.session_state['prediction_result']['key'], first)
        self.assertTrue(any('false alarm' in x.value.lower() for x in ui.warning))

    def test_simulation_play_pause_review_reset_and_tab_switch(self):
        ui = AppTest.from_file(str(ROOT / 'demo/app.py'), default_timeout=40).run()
        ui.session_state['demo_view'] = 'Workplace simulation'
        ui.run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['sim_cursor'], 0)
        ui.button(key='sim_next').click().run()
        self.assertEqual(ui.session_state['sim_cursor'], 1)
        ui.button(key='sim_next').click().run()
        self.assertEqual(ui.session_state['sim_cursor'], 2)
        ui.button(key='review_window-02').click().run()
        self.assertIn('window-02', ui.session_state['sim_reviewed'])
        ui.button(key='sim_play').click().run()
        self.assertFalse(ui.exception)
        self.assertTrue(ui.session_state['sim_playing'])
        ui.button(key='sim_play').click().run()
        self.assertFalse(ui.session_state['sim_playing'])
        before = ui.session_state['sim_cursor']
        ui.run()
        self.assertEqual(ui.session_state['sim_cursor'], before)
        ui.session_state['sim_playing'] = True
        ui.session_state['demo_view'] = 'Try a prediction'
        ui.run()
        self.assertFalse(ui.session_state['sim_playing'])
        ui.session_state['demo_view'] = 'Workplace simulation'
        ui.run()
        ui.session_state['sim_cursor'] = len(SCENES) - 1
        ui.session_state['sim_playing'] = True
        ui.session_state['sim_last_tick'] = 0
        ui.run()
        self.assertEqual(ui.session_state['sim_cursor'], len(SCENES))
        self.assertFalse(ui.session_state['sim_playing'])
        self.assertTrue(ui.button(key='sim_next').disabled)
        ui.button(key='sim_reset').click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['sim_cursor'], 0)
        self.assertFalse(ui.session_state['sim_reviewed'])


if __name__ == '__main__':
    unittest.main()
