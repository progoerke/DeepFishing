'''
Classifier base class
Implements functions init, fit, predict and evaluate
'''
class Classifier_base:

	'''
	Class variable containing parameters to optimize
	together with ranges.
	E.g. hp.uniform( 'dropout', 0, 0.6)
	'''
	space = (

	)

	'''
	Constructor
	@params: list of all model parameters
	'''
	def __init__(self, params):
		pass


	'''
	Train classifier with training data.
	All required parameters for training except for data is passed to the 
	constructor. 
	@param train_imgs: training data
	@param train_labels: targets for training data
	@param validation_imgs: validation data
	@param validation_lables: targets for validation labels
	'''
	def fit(train_imgs, train_labels, validation_imgs, validation_lables):
		pass


	'''
	Predict class labels on test data
	@param test_imgs: test data
	@return predictions for test_imgs
	'''
	def predict(test_imgs):
		pass


	'''
	Evaluate classifier performance of validation data
	@param test_imgs: data
	@param test_labels: targets
	@return classification loss
	'''
	def evaluate(test_imgs, test_labels):
		pass