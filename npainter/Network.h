#pragma once

#include <vector>

#include "Neuron.h"

namespace nn
{
	class Network
	{
	public:
		Network(const std::vector<size_t>& topology);

		std::vector<double> evaluate(const std::vector<double>& inputs);
		void train(const std::vector<double>& inputs, const std::vector<double>& targets);
	private:
		std::vector<Neuron> m_inputLayer;
		std::vector<std::vector<Neuron>> m_hiddenLayers;
		std::vector<Neuron> m_outputLayer;

		double m_recentAverageError;
	};
}