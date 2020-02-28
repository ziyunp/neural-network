import numpy as np
import pickle
import math

def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        self._cache_current = np.array(1 / (1 + np.exp(-x)))
        return self._cache_current

    def backward(self, grad_z):
        # taking the element wise products
        return grad_z * self._cache_current * (1 - self._cache_current)

class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    # returns for each element in array x the maximum of the element and 0
    def forward(self, x):
        self._cache_current = np.maximum(0, x)
        return self._cache_current

    # returns grad_z with elements less than or equal to 0 set to zero
    def backward(self, grad_z):
        grad = np.array(grad_z,copy=True)
        grad[self._cache_current <= 0] = 0
        return grad

class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self._W = np.stack([xavier_init(n_out) for i in range(n_in)])
        self._W = np.asarray(self._W)
        self._b = np.zeros(n_out)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        # if len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        #     raise ValueError("Parameter x should be an array of shape (batch_size\
        #         , input_dim) with both dimensions larger than 0")
        
        assert(len(x[0]) == self.n_in)

        self._cache_current = np.transpose(x)

        Z = np.dot(x, self._W)

        for line in Z:
            line = np.add(line, self._b)
        
        assert(len(Z[0]) == self.n_out) 
        return Z

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        assert(len(grad_z[0]) == self.n_out)

        self._grad_W_current = np.dot(self._cache_current, grad_z)
        self._grad_b_current = np.dot(np.ones(self._cache_current.shape[1]), grad_z)

        grad_loss = np.dot(grad_z, np.transpose(self._W))

        assert(len(grad_loss[0]) == self.n_in)
        return grad_loss

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        self._W = np.add(self._W, np.negative(learning_rate * self._grad_W_current))
        self._b = np.add(self._b, np.negative(learning_rate * self._grad_b_current))
        
        # self._W = tmp_W
        # self._b = tmp_b

class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        if len(neurons) != len(activations):
            raise ValueError("The length of activations should be consistent \
                with neurons")

        self._layers = []
        input_n = input_dim
        for i in range(len(neurons)):
            self._layers.append(LinearLayer(input_n, neurons[i]))
            if activations[i] == "relu":
                self._layers.append(ReluLayer())
            elif activations[i] == "sigmoid":
                self._layers.append(SigmoidLayer())
            elif activations[i] == "identity":
                pass
            input_n = neurons[i]

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """

        # if len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        #     raise ValueError("Parameter x should be an array of shape (batch_size\
        #         , input_dim) with both dimensions larger than 0")

        layer_input = x
        layer_output = None
        for this_layer in self._layers:
            layer_output = this_layer.forward(layer_input)
            layer_input = layer_output
        return layer_output

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        layer_output = grad_z
        layer_input = None
        for this_layer in reversed(self._layers):
            layer_input = this_layer.backward(layer_output)
            layer_output = layer_input
        return layer_input

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for this_layer in self._layers:
            if isinstance(this_layer, LinearLayer):
                this_layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        print("trainer constructor")
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        self._loss_layer = None
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        # two possible loss_fun values for CrossEntropy given in this file
        elif loss_fun == "cross_entropy" or loss_fun == "bce":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            raise ValueError("Loss function must be either 'mse', 'cross_entropy' or 'bce'.")

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        order = np.arange(len(input_dataset))
        np.random.shuffle(order)
        input_dataset = input_dataset[order]
        target_dataset = target_dataset[order]
        return (input_dataset, target_dataset)

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        print("train called")
        if self._loss_layer == None:
            raise ValueError("Loss layer cannot be None")
        # if given 1-d array, convert into 2-d 
        if target_dataset.ndim == 1:
            target_dataset = np.array([[t] for t in target_dataset])

        checkDatasetsDimensions(input_dataset, target_dataset)

        for epoch in range(self.nb_epoch):
            # if shuffle_flag is True, shuffle on every epoch
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            
            # calc num of batches for the given batch_size
            n_datapoints = input_dataset.shape[0]
            n_batches = math.ceil(n_datapoints/self.batch_size)
            
            # train with each batch
            for i in range(n_batches):
                input_batch = input_dataset[i * self.batch_size : (i + 1) * self.batch_size]
                target_batch = target_dataset[i * self.batch_size : (i + 1) * self.batch_size]
                outputs = self.network.forward(input_batch)
                self._loss_layer.forward(outputs, target_batch)
                loss_grad = self._loss_layer.backward()
                self.network.backward(loss_grad)
                self.network.update_params(self.learning_rate)        


    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        print("input_dataset: ", input_dataset)
        print("target_dataset: ", target_dataset)
        print("input shape: ", input_dataset.shape)
        print("target shape: ", target_dataset.shape)
        # if given 1-d array, convert into 2-d 
        if target_dataset.ndim == 1:
            target_dataset = np.array([[t] for t in target_dataset])

        checkDatasetsDimensions(input_dataset, target_dataset)
        
        predictions = self.network.forward(input_dataset)
        print("predictions.shape: ", predictions.shape)
        loss = self._loss_layer.forward(predictions, target_dataset)
        print("loss: ", loss)
        print("loss.shape: ", loss.shape)
        return loss

class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """

        if data.size == 0:
            raise ValueError("No data in the given dataset")

        col_max = np.amax(data, axis=0)
        self.col_min = np.amin(data, axis=0)
        self.params = np.subtract(col_max, self.col_min)


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """

        if len(data.shape) != 2 or data.shape[1] != len(self.params):
            raise ValueError("Invalid dataset: input dataset should have the\
                same length on the second dimension as the dataset used to\
                    initialise the preprocessor")

        return np.divide(np.subtract(data, self.col_min), self.params)


    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """

        if len(data.shape) != 2 or data.shape[1] != len(self.params):
            raise ValueError("Invalid dataset: input dataset should have the\
                same length on the second dimension as the dataset used to\
                    initialise the preprocessor")

        return np.add(np.multiply(data, self.params), self.col_min)

def checkDatasetsDimensions(input_dataset, target_dataset):
    if input_dataset.ndim != 2:
        raise ValueError("Input dataset must have 2 dimensions")
    
    input_data_points = len(input_dataset)
    input_dim = len(input_dataset[0])
    target_data_points = len(target_dataset)
    target_dim = len(target_dataset[0])

    if (input_data_points != target_data_points):
        raise ValueError("Number of data points in input and target dataset are not consistent")
    # check that each row in a dataset has the same number of features
    for row in range(input_data_points):
        if len(input_dataset[row]) != input_dim:
            raise ValueError("Dimensions of input dataset is not consistent")
    for row in range(target_data_points):
        if len(target_dataset[row]) != target_dim:
            raise ValueError("Dimensions of target dataset is not consistent")

def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    # print(x)
    # print(y)

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)

    # print(x_train_pre)
    # print(y_train)

    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
