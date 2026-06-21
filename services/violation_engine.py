import re
import time

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from config.settings import (
    DEFAULT_HELMET_SKIN_RATIO,
    DEFAULT_PARKING_COORDS,
    DEFAULT_PARKING_VIOLATION_SECONDS,
    DEFAULT_TRIPLE_OVERLAP_RATIO,
    DEFAULT_WRONG_SIDE_MIN_MOVE,
    VIOLATION_TYPES,
)


class TrafficCentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=60):
        self.next_object_id = 1
        self.objects = {}      # ID -> centroid (cx, cy)
        self.boxes = {}        # ID -> bounding box (x1, y1, x2, y2)
        self.classes = {}      # ID -> class name
        self.disappeared = {}  # ID -> frames missing
        self.history = {}      # ID -> list of historical centroids [(cx, cy), ...]
        self.first_seen = {}   # ID -> float timestamp when first tracked
        self.last_seen = {}    # ID -> float timestamp of last detection
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box, class_name):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.boxes[obj_id] = box
        self.classes[obj_id] = class_name
        self.disappeared[obj_id] = 0
        self.history[obj_id] = [centroid]
        self.first_seen[obj_id] = time.time()
        self.last_seen[obj_id] = time.time()
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.boxes[obj_id]
        del self.classes[obj_id]
        del self.disappeared[obj_id]
        del self.history[obj_id]
        del self.first_seen[obj_id]
        del self.last_seen[obj_id]

    def update(self, rects_with_classes):
        # rects_with_classes is a list of tuples: (box, class_name)
        # box = (x1, y1, x2, y2)
        if np is None:
            return self.boxes

        if len(rects_with_classes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.boxes

        input_centroids = np.zeros((len(rects_with_classes), 2), dtype="int")
        input_boxes = []
        input_classes = []

        for i, (box, class_name) in enumerate(rects_with_classes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_boxes.append(box)
            input_classes.append(class_name)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i], input_classes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Distance matrix between existing objects and new detections
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.boxes[obj_id] = input_boxes[col]
                self.classes[obj_id] = input_classes[col]
                self.disappeared[obj_id] = 0
                self.history[obj_id].append(tuple(input_centroids[col]))
                self.last_seen[obj_id] = time.time()

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_boxes[col], input_classes[col])

        return self.boxes


class ViolationEngine:
    def __init__(
        self,
        parking_violation_seconds=DEFAULT_PARKING_VIOLATION_SECONDS,
        wrong_side_min_move=DEFAULT_WRONG_SIDE_MIN_MOVE,
        triple_overlap_ratio=DEFAULT_TRIPLE_OVERLAP_RATIO,
        helmet_skin_ratio=DEFAULT_HELMET_SKIN_RATIO,
    ):
        self.parking_violation_seconds = parking_violation_seconds
        self.wrong_side_min_move = wrong_side_min_move
        self.triple_overlap_ratio = triple_overlap_ratio
        self.helmet_skin_ratio = helmet_skin_ratio
        self.reset()

    def reset(self):
        """Clear all tracking state for a fresh video/session."""
        self.tracker = TrafficCentroidTracker()
        self.parking_states = {}       # Object ID -> {"zone_idx": int, "started_at": float}
        self.logged_violations = set() # Set of (camera_id, violation_type, object_id) to avoid duplicates

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

    def find_parking_zone(self, box, zones):
        """Return the index of the parking zone containing the centroid, if any."""
        cx, cy = self._centroid(box)
        for idx, zone in enumerate(zones):
            zx1, zy1, zx2, zy2 = zone
            if zx1 < cx < zx2 and zy1 < cy < zy2:
                return idx
        return None

    def check_parking_intersection(self, box, zone):
        """Checks if vehicle centroid lies inside the parking zone box."""
        return self.find_parking_zone(box, [zone]) is not None

    def check_triple_riding(self, motorcycle_box, person_boxes):
        """Checks if a motorcycle box has 3 or more overlapping people (Triple Riding)."""
        mx1, my1, mx2, my2 = motorcycle_box
        count = 0
        
        for p_box in person_boxes:
            px1, py1, px2, py2 = p_box
            
            # Find overlap coordinates
            ix1 = max(mx1, px1)
            iy1 = max(my1, py1)
            ix2 = min(mx2, px2)
            iy2 = min(my2, py2)
            
            # Check if they overlap
            if ix1 < ix2 and iy1 < iy2:
                overlap_area = (ix2 - ix1) * (iy2 - iy1)
                person_area = (px2 - px1) * (py2 - py1)
                
                # If person area is at least 30% inside the motorcycle box, count them as riding
                if person_area > 0 and (overlap_area / person_area) > self.triple_overlap_ratio:
                    count += 1

        return count >= 3

    def _helmet_region(self, frame, person_box, motorcycle_box=None):
        if cv2 is None or np is None or frame is None:
            return None

        px1, py1, px2, py2 = person_box
        h = max(1, py2 - py1)
        w = max(1, px2 - px1)

        head_x1 = max(0, px1)
        head_x2 = min(frame.shape[1], px2)
        head_y1 = max(0, py1)
        head_y2 = min(frame.shape[0], py1 + max(8, int(h * 0.24)))

        if motorcycle_box is not None and (head_y2 <= head_y1 or head_x2 <= head_x1):
            mx1, my1, mx2, my2 = motorcycle_box
            motor_h = max(1, my2 - my1)
            motor_w = max(1, mx2 - mx1)
            head_x1 = max(0, mx1 + int(motor_w * 0.08))
            head_x2 = min(frame.shape[1], mx2 - int(motor_w * 0.08))
            head_y1 = max(0, my1 - int(motor_h * 0.05))
            head_y2 = min(frame.shape[0], my1 + int(motor_h * 0.38))

        if head_y2 <= head_y1 or head_x2 <= head_x1:
            return None

        return frame[head_y1:head_y2, head_x1:head_x2]

    def detect_helmet_violation(self, frame, person_box, motorcycle_box=None):
        """Simple computer vision heuristic for helmet detection.
        Crops head region (top 20% of person bounding box) and checks for
        color contrast/edge density or skin vs solid helmet textures.
        """
        if cv2 is None or np is None or frame is None:
            return False

        head_crop = self._helmet_region(frame, person_box, motorcycle_box=motorcycle_box)
        if head_crop is None or head_crop.size == 0:
            return False

        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.countNonZero(mask)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]
        
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        # Heuristic: If head crop contains more than 15% skin color, there is a high likelihood
        # of NO helmet (since helmets cover the head with solid plastic, visor, etc.)
        # Additionally, if hair is black/dark, we can check dark color percentages.
        # But skin ratio is a very clean heuristic!
        return skin_ratio > self.helmet_skin_ratio

    def check_wrong_side(self, obj_id, history, direction_settings, frame_width=800):
        """Checks if the vehicle is moving in the wrong direction.
        direction_settings is a dict: {
            'road_type': 'one-way'/'two-way',
            'allowed_dir': 'up'/'down',          # Used for one-way
            'left_allowed_dir': 'up'/'down',     # Used for two-way left lane
            'right_allowed_dir': 'up'/'down'     # Used for two-way right lane
        }
        """
        if len(history) < 5:
            return False

        road_type = direction_settings.get("road_type", "one-way")
        
        # Get first and last tracking points
        first_y = history[0][1]
        last_y = history[-1][1]
        diff_y = last_y - first_y
        
        if abs(diff_y) < self.wrong_side_min_move:
            return False # Vehicle hasn't moved enough
            
        # Determine allowed direction depending on lane
        if road_type == "two-way":
            cx = history[-1][0]
            width_half = frame_width // 2
            if cx < width_half:
                allowed_dir = direction_settings.get("left_allowed_dir", "up")
            else:
                allowed_dir = direction_settings.get("right_allowed_dir", "down")
        else:
            allowed_dir = direction_settings.get("allowed_dir", "down")
            
        # Check violation
        if allowed_dir == "down" and diff_y < -self.wrong_side_min_move:  # Moving UP but allowed is DOWN
            return True
        elif allowed_dir == "up" and diff_y > self.wrong_side_min_move:   # Moving DOWN but allowed is UP
            return True
            
        return False

    def check_stop_line_violation(self, box, stop_line_y, signal_state):
        """Checks if a vehicle's front edge crosses the stop-line when signal is Red."""
        if signal_state != "RED":
            return False
            
        x1, y1, x2, y2 = box
        # Assuming traffic flows top-to-bottom, vehicle front is y2
        # If flowing bottom-to-top, front is y1. Let's check both or check crossing.
        # Let's say if vehicle bounding box crosses the stop_line_y line:
        return y1 < stop_line_y < y2

    def process_violations(self, frame, detections, camera_id, camera_location, parking_zones=None, 
                           direction_settings=None, stop_line_settings=None, signal_state="GREEN",
                           custom_datetime_str=None):
        """Evaluates all rules on the current frame's detections.
        Returns a list of violations detected in this frame.
        """
        if cv2 is None or np is None:
            return []

        violations = []
        
        # Prepare list for tracker (box, class_name)
        tracker_input = []
        person_boxes = []
        motorcycle_boxes = []
        vehicle_boxes = []
        
        for d in detections:
            box = d["box"]
            cls = d["class_name"]
            
            if cls == "person":
                person_boxes.append(box)
            elif cls == "motorcycle":
                motorcycle_boxes.append(box)
                tracker_input.append((box, cls))
            elif cls in ["car", "truck", "bus"]:
                vehicle_boxes.append(box)
                tracker_input.append((box, cls))

        # Update centroid tracker
        tracked_boxes = self.tracker.update(tracker_input)
        
        # 1. Check Illegal Parking
        zones = DEFAULT_PARKING_COORDS if parking_zones is None else parking_zones
        for obj_id, box in tracked_boxes.items():
            cls = self.tracker.classes[obj_id]
            if cls in ["car", "truck", "bus"]:
                zone_idx = self.find_parking_zone(box, zones)
                if zone_idx is None:
                    if obj_id in self.parking_states:
                        del self.parking_states[obj_id]
                    continue

                now = time.time()
                parking_state = self.parking_states.get(obj_id)
                if parking_state is None or parking_state["zone_idx"] != zone_idx:
                    self.parking_states[obj_id] = {"zone_idx": zone_idx, "started_at": now}
                    continue

                parked_duration = now - parking_state["started_at"]
                if parked_duration >= self.parking_violation_seconds:
                    v_key = (camera_id, VIOLATION_TYPES["ILLEGAL_PARKING"], obj_id)
                    if v_key not in self.logged_violations:
                        violations.append({
                            "object_id": obj_id,
                            "box": box,
                            "type": VIOLATION_TYPES["ILLEGAL_PARKING"],
                            "confidence": 0.85,
                            "description": f"Vehicle parked illegally in Zone {zone_idx+1}"
                        })
                        self.logged_violations.add(v_key)

        # 2. Check Triple Riding & Helmet Violations
        for m_box in motorcycle_boxes:
            # Check Triple riding first
            is_triple = self.check_triple_riding(m_box, person_boxes)
            
            # Find matching rider overlapping motorcycle
            mx1, my1, mx2, my2 = m_box
            riders = []
            for p_box in person_boxes:
                px1, py1, px2, py2 = p_box
                # Overlap check
                ix1 = max(mx1, px1)
                iy1 = max(my1, py1)
                ix2 = min(mx2, px2)
                iy2 = min(my2, py2)
                if ix1 < ix2 and iy1 < iy2:
                    riders.append(p_box)
            
            # Check Helmets for each rider
            no_helmet_detected = False
            for rider in riders:
                if self.detect_helmet_violation(frame, rider, motorcycle_box=m_box):
                    no_helmet_detected = True
                    break
            
            # Register Helmet violation
            if no_helmet_detected:
                # Find matching obj_id in tracker for this motorcycle
                obj_id = 999  # Fallback
                for oid, obox in tracked_boxes.items():
                    if abs(obox[0] - mx1) < 10 and abs(obox[1] - my1) < 10:
                        obj_id = oid
                        break
                        
                v_key = (camera_id, VIOLATION_TYPES["HELMET"], obj_id)
                if v_key not in self.logged_violations:
                    violations.append({
                        "object_id": obj_id,
                        "box": m_box,
                        "type": VIOLATION_TYPES["HELMET"],
                        "confidence": 0.78,
                        "description": "Rider detected riding motorcycle without helmet"
                    })
                    self.logged_violations.add(v_key)

            # Register Triple Riding violation
            if is_triple:
                obj_id = 999  # Fallback
                for oid, obox in tracked_boxes.items():
                    if abs(obox[0] - mx1) < 10 and abs(obox[1] - my1) < 10:
                        obj_id = oid
                        break
                v_key = (camera_id, VIOLATION_TYPES["TRIPLE_RIDING"], obj_id)
                if v_key not in self.logged_violations:
                    violations.append({
                        "object_id": obj_id,
                        "box": m_box,
                        "type": VIOLATION_TYPES["TRIPLE_RIDING"],
                        "confidence": 0.90,
                        "description": "Motorcycle detected with triple riders"
                    })
                    self.logged_violations.add(v_key)

        # 3. Check Wrong-side Driving
        if direction_settings:
            for obj_id, box in tracked_boxes.items():
                history = self.tracker.history[obj_id]
                frame_width = frame.shape[1] if frame is not None else 800
                if self.check_wrong_side(obj_id, history, direction_settings, frame_width):
                    v_key = (camera_id, VIOLATION_TYPES["WRONG_SIDE"], obj_id)
                    if v_key not in self.logged_violations:
                        violations.append({
                            "object_id": obj_id,
                            "box": box,
                            "type": VIOLATION_TYPES["WRONG_SIDE"],
                            "confidence": 0.82,
                            "description": "Vehicle moving against allowed traffic flow direction"
                        })
                        self.logged_violations.add(v_key)

        # 4. Check Signal Violation / Stop Line Violation
        if stop_line_settings and signal_state == "RED":
            stop_line_y = stop_line_settings.get("stop_line_y", 300)
            for obj_id, box in tracked_boxes.items():
                if self.check_stop_line_violation(box, stop_line_y, signal_state):
                    v_key = (camera_id, VIOLATION_TYPES["SIGNAL_VIOLATION"], obj_id)
                    if v_key not in self.logged_violations:
                        violations.append({
                            "object_id": obj_id,
                            "box": box,
                            "type": VIOLATION_TYPES["SIGNAL_VIOLATION"],
                            "confidence": 0.88,
                            "description": "Vehicle jumped red light crossing stop-line"
                        })
                        self.logged_violations.add(v_key)

        return violations
