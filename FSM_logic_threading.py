import time
import datetime
import cv2 as cv
import numpy as np
import random
import statistics
import threading
from queue import Queue

import serial.tools.list_ports
from transitions import Machine

from ultralytics import YOLO
from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate

from skopt import Optimizer
from skopt.space import Real


class FormulationFSM:
    states = ['initialise', 'set_parameters', 'start_experiment', 'monitor_phases',
              'evaluate_stability', 'update_model', 'early_terminate', 'terminate']

    def __init__(self):
        # CONSTANTS
        MODEL_PATH = "" #ADD MODEL PATH
        MAX_ITERATIONS = 5

        self.machine = Machine(model=self, states=FormulationFSM.states, initial='initialise') # Creates FSM
        self.iteration = 0 # No of optimisation cycles
        self.max_iterations = MAX_ITERATIONS
        self.best_score = -1
        self.current_params = {}

        self.model = YOLO(MODEL_PATH) # loads trained YOLO model
        self.plate = self.find_hotplate_port()[1] # Connects to IKA hotplate
        
        self.stable_heights_log = []
        self.stable_check_count = 0

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.camera_running = False
        self.camera_thread = None

        # Start camera thread
        self.default_filename = 'Testing_output.avi'
        self.start_camera_thread(self.default_filename)

        # Search space and BO optimiser
        self.optimiser = Optimizer(
            dimensions=[Real(0, 40, name='temperature'),
                        Real(0, 1000, name='rpm'),
                        Real(0, 10, name='stir_duration'),
                        Real(0, 20, name='heat_duration'),
                        ],
            random_state=42,
            acq_func='EI'  # Expected Improvement others: PI (probability of Improvement), LCB (Lower Confidence Bound)
        )
        self.param_history = []   # Stores suggested parameters
        self.score_history = []   # Corresponding stability scores


        # FSM Transitions
        self.machine.add_transition('initialised', 'initialise', 'set_parameters', after='set_parameters_action')
        self.machine.add_transition('parameters_set', 'set_parameters', 'start_experiment', after='start_experiment_action')
        self.machine.add_transition('experiment_started', 'start_experiment', 'monitor_phases', after='monitor_phases_action')
        self.machine.add_transition('stable_detected', 'monitor_phases', 'evaluate_stability', after='evaluate_stability_action')
        self.machine.add_transition('unstable_or_timeout', 'monitor_phases', 'early_terminate', after='early_terminate_action')
        self.machine.add_transition('evaluation_done', ['evaluate_stability', 'early_terminate'], 'update_model', after='update_model_action')
        self.machine.add_transition('model_updated', 'update_model', 'set_parameters', conditions='not_converged', after='set_parameters_action')
        self.machine.add_transition('converged', 'update_model', 'terminate', after='terminate_action')

    def find_hotplate_port(self):
        """
        Scans available COM ports, tries to connect, and returns working port for IKA hotplate.
        
        Returns:
            tuple: Port device name and connected plate object
        """
        ports = serial.tools.list_ports.comports()

        for port in ports:
            print(f"Port: {port.device}, Description: {port.description}")
            # On Windows, ports are named like COMx
            if port.device.startswith('COM'):
                print(f"Trying port: {port.device} ({port.description})")
                try:
                    serial_port = port.device
                    # Create the hotplate instance
                    plate = RCTDigitalHotplate(
                        device_name="IKA RCT Digital",
                        connection_mode="serial",
                        address=None,
                        port=serial_port
                    )

                    # Establish the connectionn and initialise
                    plate.connect()
                    plate.initialize_device()

                except:
                    print("No IKA hotplate on this serial port found")

        return port.device, plate
    
    def start_camera_thread(self,filename):
        if self.camera_running:
            return
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self.camera_loop, args=(filename,), daemon=True)
        self.camera_thread.start()

    def camera_loop(self, filename):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return

        # Define video writer
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 10
        frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = cv.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        self.camera_running = True
        print("Recording live YOLO feed")

        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.65, iou=0.3, device='cpu', verbose=False)
            boxes = results[0].boxes

            # Draw YOLO results on frame, debug print
            annotated_frame = results[0].plot()
            for box in results[0].boxes:
                print(box)
                print(f"Label: Class: {box.cls.item()}, Conf: {box.conf.item():.2f}, Bbox: {box.xyxy.tolist()}")
                    
            out.write(annotated_frame)

            # Store the latest annotated frame thread safely for main loop access
            with self.frame_lock:
                self.latest_frame = annotated_frame.copy()

            cv.imshow('Live YOLO Feed', annotated_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.camera_running = False
                break

        self.cap.release()
        out.release()
        cv.destroyAllWindows()
        print("Live YOLO recording finished.")
        

    def stop_camera_thread(self):
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join()
        self.cap.release()

    def capture_image(self, filename):
        """
        Save the latest frame captured by background camera thread.
        
        Args:
            filename (str): Path to save the captured image
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                cv.imwrite(filename, self.latest_frame)
            else:
                print("No frame available yet, saving black image.")
                black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv.imwrite(filename, black_img)

    def set_parameters_action(self):
        """Suggests next parameter set using Bayesian Optimisation"""
        suggestion = self.optimiser.ask()

        self.current_params = {
            'temperature': round(suggestion[0], 2),
            'rpm': round(suggestion[1], 0),
            'stir_duration': round(suggestion[2], 0),
            'heat_duration': round(suggestion[3], 0)
        }
        print(f"Set parameters: {self.current_params}")

        self.param_history.append(suggestion)  # Save for updating BO
        self.parameters_set()

    def start_experiment_action(self):
        """
        Sends the chosen parameters to the IKA hotplate to start heating/stirring
        """
        print("Starting experiment: ")

        if self.plate:
            # Heating if temperature > 0
            if self.current_params['temperature'] > 0:
                self.plate.set_temperature(self.current_params['temperature'])
                self.plate.start_temperature_regulation()
                print(f"Heating for {self.current_params['heat_duration']} min")
                time.sleep(self.current_params['heat_duration'] * 60)  # convert to seconds
                self.plate.stop_temperature_regulation()  # turn off heating after duration

            # Stirring if rpm > 0
            if self.current_params['rpm'] > 0:
                self.plate.set_speed(self.current_params['rpm'])
                self.plate.start_stirring()
                print(f"Stirring for {self.current_params['stir_duration']} min")
                time.sleep(self.current_params['stir_duration'] * 60)  # convert to seconds
                self.plate.set_speed(0)  # stop stirring

        self.experiment_started()


    def monitor_phases_action(self):
        """
        Performs long-term monitoring (up to 24 hours) of phases using YOLO  until stability is
        confirmed or timeout is reached.
        """
        print("Starting long-term monitoring: ")

        TOTAL_MONITORING_TIME_HOURS = 24 # Max duration 
        CHECK_INTERVAL_FAST_MIN = 5
        CHECK_INTERVAL_SLOW_MIN = 30
        STABILITY_THRESHOLD = 0.05 # Height tolerance (5%)
        REQUIRED_STABLE_CHECKS = 3 # consecutive checks for stability
        EXTRA_WINDOW_CHECKS = 3 # Checks needed for FINAL stability

        total_monitoring_time_min = TOTAL_MONITORING_TIME_HOURS * 60
        elapsed_time = 0
        current_interval = CHECK_INTERVAL_FAST_MIN
        stable_count = 0 # No consecutive stable phase checks
        is_tentatively_stable = False

        prev_num_phases = None
        prev_heights = None

        # Logging setup
        log_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"stability_log_{log_start}.txt"

        time.sleep(2) # delay to allow current frame to be filled
        experiment_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(log_file, "w") as log:
            log.write("Time(min)\tNumPhases\tHeights\tStatus\n")

            while elapsed_time < total_monitoring_time_min:
                # Captures image and analyse using model
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"frame_{timestamp}.png"
                self.capture_image(img_filename)
                num_phases, heights = self.analyse_image(img_filename)

                # Compare phase cnfiguration
                if prev_num_phases is not None:
                    if self.configuration_is_same(prev_num_phases, prev_heights, num_phases, heights, STABILITY_THRESHOLD):
                        stable_count += 1
                    else:
                        stable_count = 0
                else:
                    stable_count = 0

                # if 3 stable checks in a row, the system is tentatively stable and switches to slower checking interval
                if not is_tentatively_stable and stable_count >= REQUIRED_STABLE_CHECKS:
                    is_tentatively_stable = True
                    print(f"Tentative stability reached at {elapsed_time} min.")
                    current_interval = CHECK_INTERVAL_SLOW_MIN
                    self.stable_check_count = 0
                    self.stable_heights_log = []  # Reset

                # Extra checks after tentatively stable
                if is_tentatively_stable:
                    self.stable_heights_log.append(heights)
                    self.stable_check_count += 1

                    if stable_count >= REQUIRED_STABLE_CHECKS + EXTRA_WINDOW_CHECKS:
                        print(f"Final stability confirmed at {elapsed_time} min. Stopping early.")
                        log.write(f"{elapsed_time}\t{num_phases}\t{heights}\tFinal Stable\n")
                        self.stable_detected()
                        return
                
                # Logs system stability and updates prev values
                status = "Tentatively Stable" if is_tentatively_stable else "Unstable"
                log.write(f"{elapsed_time}\t{num_phases}\t{heights}\t{status}\n")
                log.flush()

                prev_num_phases = num_phases
                prev_heights = heights

                # Waits for the next scheduled check.
                time.sleep(current_interval * 60) # seconds
                elapsed_time += current_interval

        print("Monitoring ended, final stability not confirmed.")
        self.unstable_or_timeout()

        # Stop video recording
        self.stop_camera_thread()

    def analyse_image(self, filepath):
        """
        Uses YOLO model to detect phases and extract heights of each bounding box
        
        Args:
            filepath (str): Path to image file
        
        Returns:
            tuple: (Number of detected phases, list of normalised vector positions)
        """

        results = self.model.predict(source=filepath, conf=0.65, iou=0.3, device="cpu", verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()
        heights = []
        stirrer_detected = 0

        # Extarct and normalise the vertical centre 
        for box in detections:
            y1, y2 = box[1], box[3]
            center_y = (y1 + y2) / 2
            height_norm = center_y / 480
            heights.append(round(height_norm, 3))

            if 'stirring cylinder' in box:
                stirrer_detected +=1

        heights.sort()

        num_phases = len(heights) - (stirrer_detected)

        return num_phases, heights

    def configuration_is_same(self, prev_num, prev_heights, curr_num, curr_heights, threshold=0.05):
        """Compares two configurations to check if they are within a stablity threshold.
        
        Args:
            prev_num (int): Previous number of phases
            prev_heights (list): Previous heights
            curr_num (int): Current number of phases
            curr_heights (list): Current heights
            threshold (float): Tolerance for change

        Returns:
            bool: True if configuration is stable
        
        """
        if prev_num != curr_num or len(prev_heights) != len(curr_heights):
            return False
        
        return all(abs(a - b) < threshold for a, b in zip(prev_heights, curr_heights))

    def evaluate_stability_action(self):
        """
        Calculates a wighted stability score: Based on standard deviation of heights across stable period
        and number of stable checks.
        """

        print("\nEvaluating stability:")

        if not self.stable_heights_log or self.stable_check_count == 0:
            print("No stability data captured.")
            score = 0.0
        else:
            # Flatten all heights from each check
            all_heights = [h for heights in self.stable_heights_log for h in heights]

            # Standard deviation score
            if len(all_heights) > 1:
                std_dev = statistics.stdev(all_heights)
            else:
                std_dev = 0.0

            max_std_allowed = 0.05
            score_std = max(0.0, 1.0 - (std_dev / max_std_allowed))

            # Stable check count score
            max_checks_considered = 6  # You can tune this
            score_checks = min(1.0, self.stable_check_count / max_checks_considered)

            # Combined weighted average
            weight_std = 0.6
            weight_checks = 0.4

            score = (weight_std * score_std) + (weight_checks * score_checks)
            score = round(score, 3)

            print(f"Std Dev: {std_dev:.4f}, Score from Std: {score_std:.3f}, Score from Checks: {score_checks:.3f}")

        print(f"Final Stability Score: {score}")
        self.best_score = max(self.best_score, score)
        self.evaluation_done()

    def early_terminate_action(self):
        """
        If the system never stabilizes after max duration or manually overridden.
        """
        print("Experiment unstable, early termination.")
        self.evaluation_done()

    def update_model_action(self):
        """
        Updates the Bayesian Optimisation model with new result
        """
        print("Updating model with new observation:")
        last_params = self.param_history[-1]
        last_score = self.best_score  # Assumes latest score is the one just computed
        
        self.score_history.append(last_score)
        self.optimiser.tell(last_params, last_score)

        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.converged()
        else:
            self.model_updated()

    def not_converged(self):
        """
        Checks if max optimisation cycles have been reached.
        
        Returns:
            bool: True if further optimisation is needed
        """
        return self.iteration < self.max_iterations

    def terminate_action(self):
        """
        Final cleanup method and resulting stability score.
        """
        print(f"FSM complete. Best stability score: {self.best_score:.2f}")
        if self.plate:
            self.plate.start_temperature_regulation(0)
            self.plate.speed(0)
            self.plate.disconnect()


if __name__ == "__main__":
    fsm = FormulationFSM()

    # Runs live 
    fsm.start_camera_thread('OUPUT PATH') #ADD OUTPUT PATH HERE

    fsm.initialised() # triggers first state
    while fsm.state != 'terminate':
        time.sleep(0.1)

