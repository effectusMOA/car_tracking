from BeaconDetector import BeaconDetector
from VisionModel import VisionModel
from GyroAnalysisModel import GyroAnalysisModel
from NotifyModel import NotifyModel
from KBController import KBController
from Controller import Controller
from socketlib import socketlib

import queue
from Msg import Msg

# Main execution
def main():
    bot_token = "6946139693:AAEiKB-62ra1fUvgWvvkJvhaGdU0VSxxdEU"
    kakao_api_key = "f62b8fa30e795fb79170e61e9efbccd6"
    chat_id = "834862240"

    contorller_queue = queue.Queue()
    vision_queue = queue.Queue()
    gyro_analysis_queue = queue.Queue()
    notify_queue = queue.Queue()
    main_queue = queue.Queue()

    # Initialize the models
    controller = Controller(kakao_api_key, main_queue, contorller_queue)
    kb_controller = KBController(main_queue, None)
    vision_model = VisionModel(main_queue, vision_queue)
    notify_model = NotifyModel(bot_token, None, notify_queue)
    beacon_detector_model = BeaconDetector(main_queue, None)
    gyro_analysis_model = GyroAnalysisModel(main_queue, gyro_analysis_queue)
    server_model = socketlib("server", "0.0.0.0", 5050, main_queue, None)
    rfid_server_model = socketlib("server", "0.0.0.0", 4242, main_queue, None)

    # add chat id
    notify_model.add_chat_id(chat_id)

    # Start the threads
    controller.start()
    kb_controller.start()
    vision_model.start()
    gyro_analysis_model.start()
    notify_model.start()
    beacon_detector_model.start()
    server_model.start()
    rfid_server_model.start()
    running = True

    while (running):
        while running and not main_queue.empty():
            message: Msg = main_queue.get()
            print(message)

            if (message.msg == "stop"):
                controller.stop()
                kb_controller.stop()
                vision_model.stop()
                gyro_analysis_model.stop()
                notify_model.stop()
                beacon_detector_model.stop()
                server_model.stop()
                rfid_server_model.stop()
                running = False
            elif (message.msg_to == "VisionModel"):
                vision_queue.put(message)
            elif (message.msg_to == "NotifyModel"):
                notify_queue.put(message)
            elif (message.msg_to == "GyroAnalysis"):
                gyro_analysis_queue.put(message)
            elif (message.msg_to == "Controller"):
                contorller_queue.put(message)

    # Wait for threads to finish
    controller.join()
    kb_controller.join()
    vision_model.join()
    gyro_analysis_model.join()
    notify_model.join()
    beacon_detector_model.join()
    server_model.join()
    rfid_server_model.join()

    print("Program has been terminated.")

# Run the main function
main()
