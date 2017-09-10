#pragma once

#include <vector>
#include <memory>

#include "ActivationFunction.h"
#include "Connection.h"

namespace nn
{
	class Neuron
	{
	public:
		Neuron(size_t inputsCount, std::shared_ptr<ActivationFunction> activationFunction) :
			m_activationFunction(activationFunction), m_inputConnections(inputsCount + 1)
		{}

		double getValue() const;

		void feedForward(const std::vector<double>& inputa);
		void feedForward(const std::vector<Neuron>& previousLayer);

		void updateInputWeights();

	private:
		friend class Network;

		std::shared_ptr<ActivationFunction> m_activationFunction;
		std::vector<Connection> m_inputConnections;
		double m_value;
		double m_gradient;

		static double ETA;
		static double ALPHA;
	};
}