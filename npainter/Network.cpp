#include "Network.h"

nn::Network::Network(const std::vector<size_t>& topology) :
	m_recentAverageError(0.0)
{
	size_t layersCount = topology.size();

	if (layersCount >= 3) {
		for (auto& neuronsCount : topology) {
			if (neuronsCount == 0) {
				throw std::runtime_error("Layer can't be empty");
			}
		}
	}
	else {
		throw std::runtime_error("Network must contain input and output layers, and at least one hidden");
	}
	
	std::shared_ptr<ActivationFunction> identityActivation = std::make_shared<IdentityActivationFunction>();
	std::shared_ptr<ActivationFunction> sigmoidActivation = std::make_shared<SigmoidActivationFunction>();

	m_inputLayer = std::vector<Neuron>(topology[0], Neuron(topology[0], identityActivation));

	m_hiddenLayers.resize(layersCount - 2);
	for (size_t i = 0; i < m_hiddenLayers.size(); ++i) {
		m_hiddenLayers[i] = std::vector<Neuron>(topology[i + 1], Neuron(topology[i], sigmoidActivation));
	}

	m_outputLayer = std::vector<Neuron>(topology.back(), Neuron(topology[layersCount - 2], sigmoidActivation));
}

std::vector<double> nn::Network::evaluate(const std::vector<double>& inputs)
{
	for (size_t i = 0; i < inputs.size(); ++i) {
		m_inputLayer[i].feedForward(inputs);
	}

	std::vector<Neuron>* previousLayer = &m_inputLayer;
	for (size_t i = 0; i < m_hiddenLayers.size(); ++i) {
		std::vector<Neuron>& currentLayer = m_hiddenLayers[i];
		for (size_t j = 0; j < currentLayer.size(); ++j) {
			currentLayer[j].feedForward(*previousLayer);
		}
		previousLayer = &currentLayer;
	}

	std::vector<double> result(m_outputLayer.size());

	for (size_t i = 0; i < m_outputLayer.size(); ++i) {
		m_outputLayer[i].feedForward(*previousLayer);
		result[i] = m_outputLayer[i].getValue();
	}

	return result;
}

void nn::Network::train(const std::vector<double>& inputs, const std::vector<double>& targets)
{
	if (m_outputLayer.size() != targets.size()) {
		throw std::runtime_error("Targets layout is strange");
	}

	std::vector<double> result = evaluate(inputs);

	// calculating error
	double error = 0.0;
	for (size_t i = 0; i < result.size(); ++i) {
		error += targets[i] - result[i];
	}
	error = sqrt((error * error) / result.size());

	double recentAverageSmoothingFactor = 100.0;
	m_recentAverageError = (m_recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

	// Calculating output gradients
	for (size_t i = 0; i < m_outputLayer.size(); ++i) {
		Neuron& neuron = m_outputLayer[i];
		double delta = targets[i] - neuron.m_value;
		neuron.m_gradient = delta * neuron.m_activationFunction->derivative(neuron.m_value);
	}

	// Calculating hiddent layers gradients
	std::vector<Neuron>* nextLayer = &m_outputLayer;
	for (size_t i = 0; i < m_hiddenLayers.size(); ++i) {
		std::vector<Neuron>& currentLayer = m_hiddenLayers[m_hiddenLayers.size() - 1 - i];

		for (size_t j = 0; j < currentLayer.size(); ++j) {
			Neuron& neuron = currentLayer[j];

			double dow = 0.0;
			for (size_t k = 0; k < nextLayer->size(); ++k) {
				dow += (*nextLayer)[k].m_gradient * (*nextLayer)[k].m_inputConnections[j].getWeight();
			}

			neuron.m_gradient = dow * neuron.m_activationFunction->derivative(neuron.m_value);
		}

		nextLayer = &currentLayer;
	}

	// Calculating input layer gradients
	for (size_t i = 0; i < m_inputLayer.size(); ++i) {
		Neuron& neuron = m_inputLayer[i];

		double dow = 0.0;
		for (size_t j = 0; j < nextLayer->size(); ++j) {
			dow += (*nextLayer)[j].m_gradient * (*nextLayer)[j].m_inputConnections[i].getWeight();
		}

		neuron.m_gradient = dow * neuron.m_activationFunction->derivative(neuron.m_value);
	}
}
