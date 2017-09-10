#include "Neuron.h"

double nn::Neuron::ETA = 0.15; // overall net learning rate
double nn::Neuron::ALPHA = 0.5;

double nn::Neuron::getValue() const
{
	return m_value;
}

void nn::Neuron::feedForward(const std::vector<double>& inputs)
{
	if (inputs.size() + 1 != m_inputConnections.size()) {
		throw std::runtime_error("Inputs strange layout");
	}

	m_value = 0.0;

	for (size_t i = 0; i < inputs.size(); ++i) {
		m_value += m_inputConnections[i].getWeight() * inputs[i];
	}

	m_value += m_inputConnections.back().getWeight();

	m_value = m_activationFunction->evaluate(m_value);
}

void nn::Neuron::feedForward(const std::vector<Neuron>& previousLayer)
{
	if (previousLayer.size() + 1 != m_inputConnections.size()) {
		throw std::runtime_error("Previous layer strange layout");
	}

	m_value = 0.0;

	for (size_t i = 0; i < previousLayer.size(); ++i) {
		m_value += m_inputConnections[i].getWeight() * previousLayer[i].getValue();
	}

	m_value += m_inputConnections.back().getWeight();

	m_value = m_activationFunction->evaluate(m_value);
}

void nn::Neuron::updateInputWeights()
{
	for (size_t i = 0; i < m_inputConnections.size(); ++i)
	{
		double oldDeltaWeight = m_inputConnections[i].getDeltaWeight();

		double newDeltaWeight = ETA * m_value * m_gradient + ALPHA * oldDeltaWeight;

		m_inputConnections[i].setDeltaWeight(newDeltaWeight);
		m_inputConnections[i].setWeight(m_inputConnections[i].getWeight() + newDeltaWeight);
	}
}
