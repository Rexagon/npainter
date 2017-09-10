#pragma once

namespace nn
{
	class Connection
	{
	public:
		Connection();
		Connection(double weight);

		void setWeight(double weight);
		double getWeight() const;

		void setDeltaWeight(double deltaWeight);
		double getDeltaWeight() const;

	private:
		double m_weight;
		double m_deltaWeight;
	};
}