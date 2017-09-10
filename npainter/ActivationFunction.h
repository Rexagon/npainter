#pragma once

#include <cmath>

namespace nn
{
	// Abstract activation function
	class ActivationFunction
	{
	public:
		enum class Type
		{
			Identity,
			Sigmoid,
			Tanh,
			Hlim,
			ReLU
		};

		ActivationFunction(Type type) :
			m_type(type)
		{}

		virtual ~ActivationFunction() {}

		virtual double evaluate(double x) const = 0;

		virtual double derivative(double x) const = 0;

		Type getType() const { return m_type; }

	protected:
		Type m_type;
	};

	// Identity activation function
	class IdentityActivationFunction : public ActivationFunction
	{
	public:
		IdentityActivationFunction() :
			ActivationFunction(Type::Identity)
		{}

		double evaluate(double x) const override
		{
			return x;
		}

		double derivative(double x) const override
		{
			return 1.0;
		}
	};

	// Sigmoid activation function
	class SigmoidActivationFunction : public ActivationFunction
	{
	public:
		SigmoidActivationFunction() :
			ActivationFunction(Type::Sigmoid)
		{}

		double evaluate(double x) const override
		{
			return 1.0 / (1.0 + std::exp(-x));
		}

		double derivative(double x) const override
		{
			double sigma = evaluate(x);
			return sigma * (1.0 - sigma);
		}
	};

	// Hyperbolic tangent activation function
	class TanhActivationFunction : public ActivationFunction
	{
	public:
		TanhActivationFunction() :
			ActivationFunction(Type::Tanh)
		{}

		double evaluate(double x) const override
		{
			return std::tanh(x);
		}

		double derivative(double x) const override
		{
			return 1.0 - std::pow(std::tanh(x), 2.0);
		}
	};

	// Hlim activation function
	class HlimActivationFunction : public ActivationFunction
	{
	public:
		HlimActivationFunction() :
			ActivationFunction(Type::Hlim)
		{}

		double evaluate(double x) const override
		{
			if (x > 0) {
				return 1.0;
			}
			else {
				return 0.0;
			}
		}

		double derivative(double x) const override
		{
			return 1.0;
		}
	};

	// ReLU activation function
	class ReluActivationFunction : public ActivationFunction
	{
	public:
		ReluActivationFunction() :
			ActivationFunction(Type::ReLU)
		{}

		double evaluate(double x) const override
		{
			if (x > 0.0) {
				return x;
			}
			else {
				return 0.0;
			}
		}

		double derivative(double x) const override
		{
			if (x > 0.0) {
				return 1.0;
			}
			else {
				return 0.0;
			}
		}
	};
}