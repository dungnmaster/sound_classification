import os
import time
import logging
from datetime import datetime
from queue import Empty, Queue
from threading import Thread

# from audio_classifier import AudioClassifier
from classify import NNCLassifier

SLEEP_INTERVAL = 10
AUDIO_SAMPLES_DIR = 'samples'

logging.basicConfig(
    filename='file_watcher.log',
    filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
    level=logging.INFO
)


def producer(queue):
    current_hour = datetime.now().strftime('%Y%m%d%H')
    queued_file_set = set()

    while True:
        search_dir = os.path.join(AUDIO_SAMPLES_DIR, current_hour)

        if not os.path.isdir(search_dir):
            logging.info(f'No new audio samples. Sleeping for {SLEEP_INTERVAL} seconds.')
            time.sleep(SLEEP_INTERVAL)
            continue

        files = {
            os.path.join(search_dir, f) for f in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, f))
        }
        new_files = files - queued_file_set

        for file in new_files:
            queue.put(file, block=False)
            queued_file_set.add(file)

        hour = datetime.now().strftime('%Y%m%d%H')

        if hour != current_hour:
            current_hour = hour
            queued_file_set = set()

        time.sleep(SLEEP_INTERVAL)
        logging.info(f'Enqueued {len(new_files)} new file(s) for processing. Sleeping for {SLEEP_INTERVAL} seconds.')


def consumer(queue):
    # classifier = AudioClassifier('model/audio_classifier')
    classifier = NNCLassifier('model/model_v4.h5')

    while True:
        try:
            audio_sample = queue.get()
            classifier.classify(audio_sample)

        except Empty:
            continue

        else:
            logging.info(f'Processing item {audio_sample}')
            time.sleep(2)
            queue.task_done()


if __name__ == '__main__':
    process_queue = Queue(maxsize=10000)
    producer_thread = Thread(
        target=producer,
        args=(process_queue,)
    )
    producer_thread.start()

    # create a consumer thread and start it
    consumer_thread = Thread(
        target=consumer,
        args=(process_queue,),
        daemon=True
    )
    consumer_thread.start()

    # wait for all tasks to be added to the queue
    producer_thread.join()

    # wait for all tasks on the queue to be completed
    process_queue.join()
