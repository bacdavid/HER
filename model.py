import tensorflow as tf

class Model:
    def __init__(self, sess, path, restore_only=False):
        self.sess = sess
        self.path = path
        if not restore_only:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    def save_model(self, name_of_event):
        # use step as "name" in order to restore global step afterwards
        self.saver.save(sess=self.sess, save_path=self.path + '/model-' + name_of_event)
        self.saver.export_meta_graph(self.path + '/model-' + name_of_event + '.meta')
        print("saved..." + name_of_event)

    def restore_model(self):
        try:
            ckpt = tf.train.get_checkpoint_state(self.path)
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored...")
            try:
                global_step = int(ckpt.model_checkpoint_path.split("-")[1])
                print("global step restored at..." + str(global_step))
                return global_step
            except (IndexError):
                print("unable to restore global step...")
                return 0
        except (TypeError, SystemError, AttributeError):
            print("no model restored...")
            return 0