#pragma once
#include "base/math.hpp"
#include "base/main.hpp"
#include "base/random.hpp"

class Sampling
{
public:
	Sampling(size_t samplesPerDimension, FW::Vec2f middle) :
		m_samplesPerDimension(samplesPerDimension),
		m_middle(middle), 
		m_counter(0u)
	{
		m_numberOfSamples = samplesPerDimension*samplesPerDimension;
	}
	virtual ~Sampling() {}
	
	virtual FW::Vec2f getNextSamplePos() { return FW::Vec2f(); }
	virtual bool isDone() { return m_numberOfSamples == m_counter; }

protected:
	FW::Vec2f m_middle;
	FW::Vec2f m_gridSize;
	size_t m_samplesPerDimension;
	size_t m_numberOfSamples;
	size_t m_counter;
};

class MultiJitteredSamplingWithBoxFilter : public Sampling
{
public:
	MultiJitteredSamplingWithBoxFilter(size_t samplesPerDimension, const FW::Vec2f& middle, const FW::Vec2f& s) :
	Sampling(samplesPerDimension, middle)
	{
		m_random = FW::Random();
		m_gridSize = (1.f/(float)m_samplesPerDimension)*FW::Vec2f(1./s.x, 1.f/s.y);
		m_gridStartPoint = m_middle - FW::Vec2f(.5f/s.x, .5f/s.y);
	}

	virtual FW::Vec2f getNextSamplePos(float& w)
	{
		FW_ASSERT(!(isDone()) && "Sampling index error");

		float x = m_gridSize.x*(m_counter%m_samplesPerDimension);
		float y = m_gridSize.y*(m_counter/m_samplesPerDimension);
		m_counter++;
		FW::Vec2f pos = m_gridStartPoint + FW::Vec2f(m_random.getF32(x, x + m_gridSize.x), m_random.getF32(y, y + m_gridSize.y));
		FW::Vec2f d = pos - m_middle;
		w = (1.f-FW::abs(d.x))*(1.f-FW::abs(d.y));
		return pos;
	}

private:
	FW::Random m_random;
	FW::Vec2f m_gridStartPoint;
};