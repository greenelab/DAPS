import theano
from theano import tensor as T
import numpy as np

from sklearn import linear_model

# Logistic Regression for classify
# Linear Regression for prediction

# @TODO - should probably not take y_test just to be sure
def classify(x_train, x_test, y_train, y_test):
	print('LR Begin')
	n_in = x_train.shape[1]
	n_out = 1 # @TODO this shouldn't be needed because its always 1?
	rng = np.random
	training_steps = 10000

	X = T.matrix("x")
	y = T.vector("y")

	w = theano.shared(rng.randn(n_in), name="w")
	b = theano.shared(0., name="b")

	# print("initial model")
	# print(w.get_value(), b.get_value())

	p_1 = 1 / (1 + T.exp(-T.dot(X, w) - b)) 
	prediction = p_1 > 0.5
	xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
	cost = xent.mean() + 0.01 * (w ** 2).sum()
	gw, gb = T.grad(cost, [w, b])

	train = theano.function(inputs=[X,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
          allow_input_downcast=True)
	predict_class = theano.function(inputs=[X], outputs=prediction, allow_input_downcast=True)

	print('Training LR')
	for i in range(training_steps):
	    pred, err = train(x_train, y_train)

	# print("Final model:")
	# print(w.get_value(), b.get_value())
	# print("target values for D:", y_test)
	# print("prediction on D:", predict(x_test))
	print('LR prediction')
	probas_ = predict_class(x_test)
	return probas_


def predict(x_train, x_test, y_train,):
	regr = linear_model.LinearRegression()
	regr.fit(x_train, y_train)
	return(regr.predict(x_test))