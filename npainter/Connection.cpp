#include "Connection.h"

#include "Random.h"

nn::Connection::Connection()
{
	m_weight = utils::random();
}

nn::Connection::Connection(double weight) :
	m_weight(weight)
{
}

void nn::Connection::setWeight(double weight)
{
	m_weight = weight;
}

double nn::Connection::getWeight() const
{
	return m_weight;
}

void nn::Connection::setDeltaWeight(double deltaWeight)
{
	m_deltaWeight = deltaWeight;
}

double nn::Connection::getDeltaWeight() const
{
	return m_deltaWeight;
}
