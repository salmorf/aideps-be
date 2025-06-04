import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class MedicalYOLOInference:
    def __init__(self):
        self.model = None
        self.class_colors = {
            "right_breast": (255, 0, 0),
            "left_breast": (0, 255, 0),
            "right_areola": (0, 0, 255),
            "left_areola": (255, 255, 0),
            "right_nipple": (255, 0, 255),
            "left_nipple": (128, 0, 255),
            "right_jugular": (0, 128, 255),
            "left_jugular": (255, 128, 0),
        }
        self.label_translations = {
            "seno_destro": "right_breast",
            "seno_sinistro": "left_breast",
            "areola_destra": "right_areola",
            "areola_sinistra": "left_areola",
            "capezzolo_dx": "right_nipple",
            "capezzolo_sx": "left_nipple",
            "giugulare_dx": "right_jugular",
            "giugulare_sx": "left_jugular",
        }
        self.corrections_applied = 0
        self.anatomical_correction_enabled = True

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            return f"✅ Model loaded: {os.path.basename(model_path)}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def analyze_image(self, image, confidence_threshold, model_path=None):
        if self.model is None or model_path is not None:
            if model_path is None:
                return (image, "❌ No model loaded. Please load a model first.")
            self.load_model(model_path)

        if self.model is None:
            return (image, "❌ Model not loaded properly.")

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return (image, "❌ Invalid image format.")

        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        try:
            results = self.model(
                temp_path,
                conf=confidence_threshold,
                iou=0.7,
                agnostic_nms=True,
                max_det=10,
            )
        except Exception as e:
            os.remove(temp_path)
            return (image, f"❌ Inference error: {str(e)}")

        detection_info = []
        result_image = image_rgb.copy()
        self.corrections_applied = 0
        all_detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names.get(cls, f"class_{cls}")
                class_name = self.label_translations.get(class_name, class_name)
                all_detections.append(
                    {
                        "class": class_name,
                        "class_id": cls,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - intersection_area
            return intersection_area / union_area if union_area > 0 else 0

        def apply_anatomical_correction(detections):
            if not self.anatomical_correction_enabled:
                return detections
            corrected_detections = []
            img_height, img_width = image_rgb.shape[:2]
            img_center_x = img_width / 2
            correction_map = {
                "right_areola": "left_areola",
                "left_areola": "right_areola",
                "right_breast": "left_breast",
                "left_breast": "right_breast",
                "right_nipple": "left_nipple",
                "left_nipple": "right_nipple",
                "right_jugular": "left_jugular",
                "left_jugular": "right_jugular",
            }
            class_id_map = {
                id: next(
                    (
                        corr_id
                        for corr_id, corr_name in self.model.names.items()
                        if corr_name == correction_map.get(name)
                    ),
                    id,
                )
                for id, name in self.model.names.items()
                if name in correction_map
            }
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                box_center_x = (x1 + x2) / 2
                class_name = det["class"]
                class_id = det["class_id"]
                on_left_side = box_center_x < img_center_x
                should_correct = (("right" in class_name and not on_left_side)) or (
                    ("left" in class_name and on_left_side)
                )
                if should_correct and class_name in correction_map:
                    corrected_name = correction_map[class_name]
                    corrected_id = class_id_map.get(class_id, class_id)
                    det_copy = det.copy()
                    det_copy["class"] = corrected_name
                    det_copy["class_id"] = corrected_id
                    corrected_detections.append(det_copy)
                    self.corrections_applied += 1
                else:
                    corrected_detections.append(det)
            return corrected_detections

        all_detections = apply_anatomical_correction(all_detections)

        filtered_detections = []
        for det in all_detections:
            bbox = det["bbox"]
            is_duplicate = any(
                calculate_iou(bbox, existing["bbox"]) > 0.5
                for existing in filtered_detections
            )
            if not is_duplicate:
                filtered_detections.append(det)
                detection_info.append(det)

        for det in filtered_detections:
            class_name = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            color = self.class_colors.get(class_name, (192, 192, 192))
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.replace('_', ' ').title()}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        self.last_detections = filtered_detections
        img_height = result_image.shape[0]
        cv2.putText(
            result_image,
            f"Anatomical corrections applied: {self.corrections_applied}",
            (10, img_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
            False,
        )

        os.remove(temp_path)
        report = self.create_report(detection_info, confidence_threshold)
        if self.anatomical_correction_enabled:
            report += f"\nAnatomical correction applied: {self.corrections_applied}\n"
            if self.corrections_applied > 0:
                report += "Class labels have been corrected based on anatomical position within the image.\n"
        else:
            report += "\nAnatomical correction disabled.\n"

        return (result_image, report)

    def create_report(self, detections, confidence_threshold):
        if not detections:
            return "No anatomical structures detected. Try lowering the confidence threshold."

        report = f"DETECTION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"
        report += f"Confidence threshold: {confidence_threshold}\n"
        report += f"Total detections: {len(detections)}\n\n"

        by_class = {}
        for det in detections:
            class_name = det["class"]
            by_class.setdefault(class_name, []).append(det)

        for class_name in sorted(by_class.keys()):
            items = by_class[class_name]
            report += (
                f"{class_name.replace('_', ' ').title()} ({len(items)} detections):\n"
            )
            for i, det in enumerate(items, 1):
                x1, y1, x2, y2 = det["bbox"]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                conf = det["confidence"]
                center_x = (x1 + x2) / 2
                position = "left" if center_x < 320 else "right"
                report += f"{i}. Confidence: {conf:.2f}\n"
                report += f"   Dimensions: {width}x{height} px (area: {area} px²)\n"
                report += f"   Position: {position} side of the image\n"
                report += f"   Bounding Box: ({x1}, {y1}) - ({x2}, {y2})\n\n"

        report += "ADDITIONAL INFORMATION:\n"
        report += "- Anatomical correction logic was applied to ensure proper labeling of detected structures.\n"
        report += "- 'Right' anatomical structures are expected to appear on the left side of the image (observer's view).\n"
        report += "- 'Left' anatomical structures are expected to appear on the right side of the image.\n"

        return report
