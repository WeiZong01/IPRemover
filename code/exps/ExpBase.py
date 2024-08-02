from pathlib import Path
import pickle
from datetime import datetime


class ExpBase:
    def __init__(self, out_dir, exp_status_fname):
        """
        """
        self.out_dir = Path(out_dir)

        # create folder if not exists
        if not self.out_dir.exists():
            Path.mkdir(self.out_dir, parents=True)

        # load experiment status if exists
        self.exp_status_path = self.out_dir.joinpath(exp_status_fname)
        self.status_dic = {}
        if self.exp_status_path.exists():
            with open(self.exp_status_path, 'rb') as handle:
                self.status_dic = pickle.load(handle)

    def save_status(self):
        # save experiment status
        with open(self.exp_status_path, 'wb') as handle:
            pickle.dump(self.status_dic, handle)

    def save_flag(self, flag_path, flag=None):
        if flag is None:
            flag = datetime.now()

        with open(flag_path, 'wb') as handle:
            pickle.dump(flag, handle)

    def load_flag(self, flag_path):
        if not flag_path.exists():
            return None

        with open(flag_path, 'rb') as handle:
            flag = pickle.load(handle)
        return flag

    def run(self):
        """
        run the experiments
        """
        raise NotImplementedError
