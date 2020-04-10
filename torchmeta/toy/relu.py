import numpy as np

from torchmeta.utils.data import Task, MetaDataset


class Relu(MetaDataset):
    """
    Parameters
    ----------
    num_samples_per_task : int
        Number of examples per task.

    num_tasks : int (default: 2)
        Overall number of tasks to sample.

    noise_std : float, optional
        Amount of noise to include in the targets for each task. If `None`, then
        nos noise is included, and the target is either a sine function, or a
        linear function of the input.

    transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the input.

    target_transform : callable, optional
        A function/transform that takes a numpy array of size (1,) and returns a
        transformed version of the target.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    """

    def __init__(self, num_samples_per_task, num_tasks=2,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None, seed=None):
        super(Relu, self).__init__(meta_split='train',
                                   target_transform=target_transform, dataset_transform=dataset_transform,
                                   seed=seed)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform

        self._input_range = np.array([-5.0, 5.0])

        self._signs = None

    @property
    def signs(self):
        if self._signs is None:
            self._signs = np.ones((self.num_tasks,), dtype=np.int)
            self._signs[self.num_tasks // 2:] = -1
            self.np_random.shuffle(self._signs)
        return self._signs

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):

        task = ReluTask(index, self.signs[index], self._input_range,
                        self.noise_std, self.num_samples_per_task, self.transform,
                        self.target_transform, np_random=self.np_random)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class ReluTask(Task):
    def __init__(self, index, sign, input_range, noise_std,
                 num_samples, transform=None, target_transform=None,
                 np_random=None):
        super(ReluTask, self).__init__(index, None)  # Regression task
        self.sign = sign
        self.input_range = input_range
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        self._inputs = np_random.uniform(input_range[0], input_range[1],
                                         size=(num_samples, 1))
        self._targets = sign * np.maximum(self._inputs, 0)
        if (noise_std is not None) and (noise_std > 0.):
            self._targets += noise_std * np_random.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self._inputs[index], self._targets[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (input, target)
