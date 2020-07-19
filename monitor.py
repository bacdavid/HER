import tensorflow as tf
#import cv2 as cv
import numpy as np

class Monitor:
    def __init__(self, type, **kwargs):
        self.type = type
        types_available = {
            'scalar': self._setup_scalar,
            'image': self._setup_image,
            'video': self._setup_video,
            'live_video':self._setup_live_video
        }
        types_available[self.type](**kwargs)

    def record(self, **kwargs):
        self.recorder(**kwargs)

    def stop_recording(self):
        self.stopper()

    def _setup_scalar(self, sess, name_of_var):
        self.sess = sess
        self.name = name_of_var
        self.var = tf.Variable(0., trainable=False)
        self.op = tf.summary.merge(inputs=[tf.summary.scalar(self.name, self.var)])
        self.writer = tf.summary.FileWriter('./summary', self.sess.graph)
        self.recorder = self._record_scalar

    def _setup_image(self, sess, name_of_var):
        self.sess = sess
        self.name = name_of_var
        self.var = tf.Variable(0., trainable=False)
        self.op = tf.summary.merge(inputs=[tf.summary.image(self.name, self.var)])
        self.writer = tf.summary.FileWriter('./summary', self.sess.graph)
        self.recorder = self._record_image

    def _setup_video(self, shape=(80, 80), overwrite=True):
        self.shape = shape
        self.overwrite = overwrite
        self.video_number = 0
        if self.overwrite:
            self.op = cv.VideoWriter('video-0.avi', cv.VideoWriter_fourcc(*'MJPG'), 20., self.shape, isColor=True)
        self.recorder = self._record_video
        self.stopper = self._stop_video

    def _setup_live_video(self):
        self.recorder = self._record_live
        self.stopper = self._stop_live

    def _record_scalar(self, value, step):
        self.writer.add_summary(self.sess.run(self.op, feed_dict={self.var: value}), step)
        self.writer.flush()

    def _record_image(self, image, step):
        assert image.ndim == 4, "image must have to be of format (batch, h, w, channels) for image summary"
        if image.shape[-1] != 1:
            image = image[:, :, :, -2 : -1]
        self.writer.add_summary(self.sess.run(self.op, feed_dict={self.var: image}), step)
        self.writer.flush()

    def _record_video(self, screen):
        assert screen.ndim == 4, "screen must have to be of format (batch, h, w, channels) for video summary"
        if screen.shape[-1] != 1:
            screen = screen[:, :, :, -2 : -1]
        screen = screen[-1, :, :, -1]
        screen *= 255
        screen = screen.astype(np.uint8)
        if not self.overwrite:
            self.op = cv.VideoWriter('video-' + str(self.video_number) + '.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.,
                                     self.shape, isColor=True)
            self.video_number += 1
        self.op.write(cv.cvtColor(screen, cv.COLOR_GRAY2RGB))

    def _record_live(self, screen):
        assert screen.ndim == 4, "screen must have to be of format (batch, h, w, channels) for live summary"
        cv.namedWindow('live_play', cv.WINDOW_NORMAL)
        cv.imshow('live_play', screen[-1, :, :, -1])

    def _stop_video(self):
        self.op.release()

    def _stop_live(self):
        cv.destroyAllWindows()

