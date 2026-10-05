"""The operator page for the optional per-region homography mode.

The page is mostly Tk, and a test that needs a window is a test that does not run
on a build machine. What is exercised here is everything the page decides *before*
it draws: which crop region it will fit, whether the gate is open, which board the
fields describe, whether a fit may be saved, and how it grades one. Those are the
decisions that can be wrong in a way the operator would not notice -- a gate that
opens without a crop region fits a homography to a raster that does not exist, and
a save that goes through with a failed flatness check stores a homography that
measures the wrong plane for as long as nobody re-checks it.

So the page is built the way ``tests/test_crop_marking_ui.py`` builds the dashboard:
``__new__`` to skip ``__init__`` and its windows, then the attributes the method under
test actually touches. Mocks stand in for the widgets, so a missing attribute shows up
here rather than as a blank panel in front of an operator.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import region_calibration
import CalibrateAPP.region_calibration_ui as region_ui

SETTINGS = {
    "enabled": True,
    "store_file": "Files/region_homographies.json",
    "board_profiles_file": "Files/board_definitions.json",
    "default_profile": "Region board 17x7",
    "minimum_corners": 8,
    "maximum_rms_mm": 0.5,
    "maximum_tilt_degrees": 2.0,
    "maximum_gap_mm": 2.0,
    "maximum_scale_error_percent": 2.0,
    "refine_intrinsics": False,
    "minimum_refinement_views": 5,
}

MARKED = {
    "center_normalized": [0.5, 0.25],
    "width_normalized": 0.77,
    "angle_degrees": 0.0,
    "line_normalized": [[0.22, 0.28], [0.78, 0.30]],
}


def make_page(cameras=None, profiles=None, index=0):
    """A region page with no window, no camera and no executor."""
    page = region_ui.RegionCalibrationApp.__new__(region_ui.RegionCalibrationApp)
    page.crop_definitions = {
        "version": 1,
        "cameras": cameras if cameras is not None
        else {"1": [dict(MARKED), dict(MARKED)], "2": [None, None]},
    }
    page.profiles = (profiles if profiles is not None
                     else region_calibration.default_board_profiles())
    page.profile_index = index
    page.region_index = 1
    page.crop_ready = False
    page.undistorter = None
    page.pending = None
    page.capture_busy = False
    page.refinement_views = []
    page.region_status = {1: None, 2: None}
    page.camera_index = region_ui.CAMERA_IDS[0]
    page.cap = None
    page.camera_running = False
    page.contours = None

    page.profile_var = MagicMock()
    page.profile_var.get.return_value = page.profiles[index].name
    page.dictionary_var = MagicMock()
    page.dictionary_var.get.return_value = page.profiles[index].dictionary
    page.squares_x_var = MagicMock()
    page.squares_x_var.get.return_value = str(page.profiles[index].squares_x)
    page.squares_y_var = MagicMock()
    page.squares_y_var.get.return_value = str(page.profiles[index].squares_y)
    page.square_var = MagicMock()
    page.square_var.get.return_value = f"{page.profiles[index].square_length_mm:g}"
    page.marker_var = MagicMock()
    page.marker_var.get.return_value = f"{page.profiles[index].marker_length_mm:g}"
    page.refine_var = MagicMock()
    page.refine_var.get.return_value = False

    page.gate_banner = MagicMock()
    page.workspace = MagicMock()
    page.preview_title = MagicMock()
    page.region_state = MagicMock()
    page.fit_advisory = MagicMock()
    page.refine_note = MagicMock()
    page.report = MagicMock()
    page.status_banner = MagicMock()
    page.status_text = MagicMock()
    page.capture_btn = MagicMock()
    page.save_btn = MagicMock()
    page.shrink_btn = MagicMock()
    page.region_buttons = {1: MagicMock(), 2: MagicMock()}
    page.start_btn = MagicMock()
    page.stop_btn = MagicMock()
    return page


class PreviewScalingTests(unittest.TestCase):
    """The crop is shown at its own aspect ratio, never stretched or magnified."""

    def test_a_wide_crop_is_scaled_to_the_viewport_width(self):
        crop = np.zeros((552, 2208, 3), np.uint8)
        scaled = region_ui.scale_to_viewport(crop, (1104, 400))
        self.assertEqual(scaled.shape[1], 1104)
        # 2208 x 552 is 4:1, so half the width is half the height.
        self.assertEqual(scaled.shape[0], 276)

    def test_a_viewport_taller_than_the_crop_limits_on_width(self):
        crop = np.zeros((552, 2208, 3), np.uint8)
        scaled = region_ui.scale_to_viewport(crop, (600, 4000))
        self.assertEqual(scaled.shape[1], 600)
        self.assertEqual(scaled.shape[0], 150)

    def test_a_crop_smaller_than_the_viewport_is_not_magnified(self):
        """Upscaling a crop would invent detail the operator then judges."""
        crop = np.zeros((100, 400, 3), np.uint8)
        scaled = region_ui.scale_to_viewport(crop, (1920, 1080))
        self.assertIs(scaled, crop)

    def test_no_viewport_means_nothing_to_show(self):
        self.assertIsNone(
            region_ui.scale_to_viewport(np.zeros((10, 10, 3), np.uint8), None))


class GateTests(unittest.TestCase):
    """The page refuses to do anything until both crop regions exist."""

    def test_the_gate_opens_only_when_both_regions_are_marked(self):
        page = make_page()
        page._refresh_controls = MagicMock()
        page._refresh_gate()
        self.assertTrue(page.crop_ready)
        page.gate_banner.pack_forget.assert_called_once()

    def test_one_missing_region_closes_the_gate_and_names_it(self):
        page = make_page(cameras={"1": [dict(MARKED), None], "2": [None, None]})
        page._refresh_controls = MagicMock()
        page._refresh_gate()
        self.assertFalse(page.crop_ready)
        banner = page.gate_banner.configure.call_args.kwargs["text"]
        self.assertIn("REGION 02", banner)
        self.assertNotIn("REGION 01,", banner)

    @patch.object(region_ui, 'CAMERA_IDS', [2, 3])
    def test_an_unmarked_camera_closes_the_gate_for_both_regions(self):
        page = make_page(cameras={"1": [dict(MARKED), dict(MARKED)], "2": [None, None]})
        page.camera_index = region_ui.CAMERA_IDS[1]
        page._refresh_controls = MagicMock()
        page._refresh_gate()
        self.assertFalse(page.crop_ready)
        banner = page.gate_banner.configure.call_args.kwargs["text"]
        self.assertIn("REGION 01", banner)
        self.assertIn("REGION 02", banner)

    def test_a_definition_that_is_not_a_region_counts_as_missing(self):
        """A cleared region is stored as None, and a half-written one as a bare value."""
        page = make_page(cameras={"1": ["not a region", dict(MARKED)], "2": [None, None]})
        self.assertIsNone(page._crop_for(1))
        self.assertIsNotNone(page._crop_for(2))

    def test_an_absent_camera_counts_as_missing_rather_than_crashing(self):
        page = make_page(cameras={"2": [dict(MARKED), dict(MARKED)]})
        self.assertIsNone(page._crop_for(1))


class CameraNumberTests(unittest.TestCase):
    """The store is keyed by side, not by device index."""

    @patch.object(region_ui, 'CAMERA_IDS', [2, 3])
    def test_the_device_index_is_mapped_to_the_side_that_owns_it(self):
        page = make_page()
        page.camera_index = region_ui.CAMERA_IDS[0]
        self.assertEqual(page.camera_number(), 1)
        page.camera_index = region_ui.CAMERA_IDS[1]
        self.assertEqual(page.camera_number(), 2)

    def test_an_unknown_index_falls_back_to_the_first_side(self):
        page = make_page()
        page.camera_index = 99
        self.assertEqual(page.camera_number(), 1)


class BoardFieldTests(unittest.TestCase):
    """The board is whatever the fields currently say, not what the profile says."""

    def test_the_fields_describe_the_board(self):
        page = make_page()
        page.squares_x_var.get.return_value = "9"
        page.square_var.get.return_value = "12.5"
        page.marker_var.get.return_value = "9.0"
        board = page.current_profile()
        self.assertEqual((board.squares_x, board.square_length_mm), (9, 12.5))
        self.assertEqual(board.marker_length_mm, 9.0)

    def test_a_blank_name_falls_back_rather_than_saving_an_empty_profile(self):
        page = make_page()
        page.profile_var.get.return_value = "   "
        self.assertTrue(page.current_profile().name)

    def test_unparseable_numbers_become_zero_and_fail_validation(self):
        page = make_page()
        page.squares_x_var.get.return_value = "many"
        page.square_var.get.return_value = ""
        board = page.current_profile()
        self.assertEqual((board.squares_x, board.square_length_mm), (0, 0.0))
        with self.assertRaises(ValueError):
            board.validate()

    def test_the_configured_default_profile_is_the_one_selected(self):
        page = make_page()
        self.assertEqual(page.profiles[page.profile_index].name, SETTINGS["default_profile"])

    def test_an_unknown_default_falls_back_to_the_first_profile(self):
        page = region_ui.RegionCalibrationApp.__new__(
            region_ui.RegionCalibrationApp)
        page.profiles = region_calibration.default_board_profiles()
        with patch.object(region_ui, "REGION_SETTINGS",
                          dict(SETTINGS, default_profile="no such board")):
            self.assertEqual(page._default_profile_index(), 0)


class RegionStateTests(unittest.TestCase):
    """Each region reports its own state, because they are calibrated separately."""

    def test_the_states_name_the_missing_crop_the_missing_fit_and_the_saved_one(self):
        page = make_page(cameras={"1": [dict(MARKED), None], "2": [None, None]})
        page.region_status = {1: None, 2: None}
        page._refresh_region_state()
        text = page.region_state.configure.call_args.kwargs["text"]
        self.assertIn("REGION 01: not calibrated", text)
        self.assertIn("REGION 02: no crop region marked", text)

    def test_a_saved_region_reports_its_measured_numbers(self):
        page = make_page()
        page.region_status = {
            1: {"metrics": {"rms_mm": 0.123, "corners": 96},
                "plane_alignment": {"tilt_degrees": 1.25}},
            2: None,
        }
        page._refresh_region_state()
        text = page.region_state.configure.call_args.kwargs["text"]
        self.assertIn("RMS 0.123 mm", text)
        self.assertIn("96 corners", text)
        self.assertIn("tilt 1.25", text)


class LimitsTests(unittest.TestCase):
    """The grade boundaries come from config, with the documented fallbacks."""

    def test_the_configured_limits_are_used(self):
        page = make_page()
        limits = page._limits()
        self.assertEqual(limits["rms"], 0.5)
        self.assertEqual(limits["tilt"], 2.0)
        self.assertEqual(limits["scale"], 0.02)

    def test_a_missing_setting_falls_back_instead_of_raising(self):
        page = make_page()
        with patch.object(region_ui, "REGION_SETTINGS", {}):
            limits = page._limits()
        self.assertEqual(limits["rms"], 0.5)
        self.assertEqual(limits["scale"], 0.02)


class RefinementNoteTests(unittest.TestCase):
    """Refinement cannot run on one view, and the page says so instead of implying it did."""

    def test_off_by_default_with_the_reason(self):
        page = make_page()
        note = page._refinement_note()
        self.assertIn("5 views", note)
        self.assertIn("Leave this off", note)

    def test_on_but_short_of_views_reports_how_many_are_needed(self):
        page = make_page()
        page.refine_var.get.return_value = True
        page.refinement_views = [None, None]
        note = page._refinement_note()
        self.assertIn("2 view(s)", note)
        self.assertIn("5 are needed", note)


class FitReportTests(unittest.TestCase):
    """A fit that fails a gate must be marked as failing in the report the operator reads."""

    def _fit(self):
        return region_calibration.RegionFit(
            homography=np.eye(3), mm_per_pixel=0.1, rms_mm=0.05, mean_mm=0.04,
            max_mm=0.12, p95_mm=0.1,
            stats=region_calibration.FitStats(corners=96, dropped=0, spread_ratio=0.5),
            board_definition=region_calibration.REGION_BOARD_PROFILE,
            intrinsic_evaluation=region_calibration.IntrinsicEvaluation(
                rms_px=0.4, mean_px=0.3, max_px=0.9, corners=96),
            image_size=(3456, 4608), output_size=(2208, 552),
        )

    def test_a_flat_board_on_the_plane_reads_as_ok(self):
        page = make_page()
        alignment = region_calibration.PlaneAlignment(
            tilt_degrees=0.5, gap_mm=0.1, expected_gap_mm=0.0)
        agreement = region_calibration.ScaleAgreement(
            local_mm_per_pixel=0.1, reference_mm_per_pixel=0.1005)
        report = page._fit_report(self._fit(), alignment, agreement, None)
        self.assertIn("Flatness", report)
        self.assertIn("OK", report)
        self.assertNotIn("FAILS", report)

    def test_a_tilted_board_is_marked_as_failing_the_flatness_gate(self):
        page = make_page()
        alignment = region_calibration.PlaneAlignment(
            tilt_degrees=6.0, gap_mm=8.0, expected_gap_mm=0.0)
        report = page._fit_report(self._fit(), alignment, None, None)
        self.assertIn("FAILS", report)
        self.assertIn("not lying on the measurement plane", report)

    def test_a_mis_declared_board_size_is_marked_as_failing_the_cross_check(self):
        page = make_page()
        agreement = region_calibration.ScaleAgreement(
            local_mm_per_pixel=0.15, reference_mm_per_pixel=0.1)
        report = page._fit_report(self._fit(), None, agreement, None)
        self.assertIn("FAILS", report)
        self.assertIn("wrong size", report)

    def test_the_report_says_the_residual_is_not_a_guarantee(self):
        page = make_page()
        report = page._fit_report(self._fit(), None, None, None)
        self.assertIn("not a guarantee", report)


class SaveGateTests(unittest.TestCase):
    """Nothing is written without a fit, and a failed gate needs a deliberate override."""

    def test_saving_without_a_fit_does_nothing(self):
        page = make_page()
        page.pending = None
        with patch("CalibrateAPP.region_calibration_ui.messagebox") as box:
            page.save_region()
        box.askyesno.assert_not_called()
        box.showerror.assert_not_called()

    def test_a_failed_flatness_check_refuses_unless_the_operator_insists(self):
        page = make_page()
        fit = FitReportTests()._fit()
        alignment = region_calibration.PlaneAlignment(
            tilt_degrees=6.0, gap_mm=8.0, expected_gap_mm=0.0)
        page.pending = (fit, alignment, None, None, 1)
        with patch("CalibrateAPP.region_calibration_ui.messagebox") as box, \
                patch.object(region_ui, "region_store_path", return_value="nowhere"):
            box.askyesno.return_value = False
            page.save_region()
            box.askyesno.assert_called_once()
        # Declining leaves the fit pending, so the operator can re-shoot rather than
        # losing the measurement they just took.
        self.assertIsNotNone(page.pending)


class CheckRegionSelectorTests(unittest.TestCase):
    """The shared check controls switch between global and region mappings."""

    def _check_page(self):
        from CalibrateAPP import calibration_ui
        page = calibration_ui.CalibrationCheckApp.__new__(
            calibration_ui.CalibrationCheckApp)
        page.region_check_index = 1
        page.measurement_check_mode = "global"
        page.preview_region_definition = None
        page.check_preview_frozen = False
        page.processing_verification = False
        page.global_check_mode_btn = MagicMock()
        page.region_check_mode_btn = MagicMock()
        page.region_selector_row = MagicMock()
        page.region_check_buttons = {1: MagicMock(), 2: MagicMock()}
        page.region_check_info = MagicMock()
        page._refresh_region_check_info = MagicMock()
        page._refresh_stage_ui = MagicMock()
        page._hide_manual_measurement = MagicMock()
        page._set_check_report = MagicMock()
        page._set_mapping_preview_title = MagicMock()
        page._camera_number = MagicMock(return_value=1)
        page._check_region_crop = MagicMock(return_value=None)
        return page

    def test_selecting_a_region_marks_the_button_and_reselects(self):
        from CalibrateAPP import calibration_ui
        page = self._check_page()
        page.select_check_region(2)
        self.assertEqual(page.region_check_index, 2)
        page.region_check_info.configure  # touched by the info refresh
        page._refresh_region_check_info.assert_called_once()

    def test_the_check_is_not_ready_without_a_crop_region(self):
        from CalibrateAPP import calibration_ui
        page = self._check_page()
        page._check_region_crop.return_value = None
        with patch.object(calibration_ui, "CONFIG",
                          {"region_homography": dict(SETTINGS)}):
            self.assertFalse(page._region_check_ready())

    def test_the_check_is_not_ready_when_the_mode_is_off(self):
        from CalibrateAPP import calibration_ui
        page = self._check_page()
        page._check_region_crop.return_value = dict(MARKED)
        with patch.object(calibration_ui, "CONFIG",
                          {"region_homography": dict(SETTINGS, enabled=False)}):
            self.assertFalse(page._region_check_ready())

    def test_region_mode_reuses_the_normal_check_controls(self):
        from CalibrateAPP import calibration_ui
        page = self._check_page()
        with patch.object(calibration_ui, "CONFIG",
                          {"region_homography": dict(SETTINGS)}):
            page.select_measurement_check_mode("region")
        self.assertEqual(page.measurement_check_mode, "region")
        page._hide_manual_measurement.assert_called_once()
        page._refresh_stage_ui.assert_called_once()

    def test_measurement_button_routes_to_the_selected_region_mapping(self):
        page = self._check_page()
        page.measurement_check_mode = "region"
        page._verify_region_measurement_accuracy = MagicMock()
        page.verify_measurement_accuracy()
        page._verify_region_measurement_accuracy.assert_called_once()


class UnselectedCameraTests(unittest.TestCase):
    """The check page is built before a camera is picked, and the section must survive that.

    This is a regression test. The region section refreshed itself at the end of
    construction, and that refresh resolved the camera's side number from a calibration
    path -- which does not exist until a camera has been selected. ``camera_config``
    raised ``StopIteration``, it escaped the page constructor, and the whole Calibration
    Check page died with "Could not open camera tool" for every operator who had the
    optional mode switched on. The earlier tests all built the page with ``__new__`` and
    never ran the constructor, so none of them could have caught it.
    """

    def _page(self):
        from CalibrateAPP import calibration_ui
        page = calibration_ui.CalibrationCheckApp.__new__(
            calibration_ui.CalibrationCheckApp)
        page.camera_index = None
        page.region_check_index = 1
        page.region_check_info = MagicMock()
        return page

    def test_the_camera_number_falls_back_rather_than_raising(self):
        self.assertEqual(self._page()._camera_number(), 1)

    def test_there_is_no_region_to_check_without_a_camera(self):
        from CalibrateAPP import calibration_ui
        page = self._page()
        with patch.object(calibration_ui, "CONFIG",
                          {"region_homography": dict(SETTINGS)}):
            self.assertIsNone(page._check_region_crop())
            self.assertFalse(page._region_check_ready())

    def test_the_section_says_so_rather_than_naming_a_region(self):
        page = self._page()
        page._refresh_region_check_info()
        text = page.region_check_info.configure.call_args.kwargs["text"]
        self.assertIn("select a camera", text)


class CheckLivePreviewTests(unittest.TestCase):
    def test_region_mode_displays_only_the_deskewed_crop_but_keeps_the_raw_frame(self):
        from CalibrateAPP import calibration_ui
        page = calibration_ui.CalibrationCheckApp.__new__(
            calibration_ui.CalibrationCheckApp)
        page.measurement_check_mode = "region"
        page.preview_region_definition = dict(MARKED)
        page.camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        page.dist_coeffs = np.zeros(5)
        page.calibration_image_size = (640, 480)
        raw = np.zeros((480, 640, 3), np.uint8)
        source = MagicMock()
        source.read.return_value = True, raw

        returned_source, current_frame, display = page._prepare_live_preview(
            source, (800, 600))

        self.assertIs(returned_source, source)
        self.assertIs(current_frame, raw)
        self.assertEqual(display.shape[1] // display.shape[0], 4)
        self.assertLess(display.shape[0], 480)


if __name__ == "__main__":
    unittest.main()
