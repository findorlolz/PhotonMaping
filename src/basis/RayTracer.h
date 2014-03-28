#pragma once

#include "base/String.hpp"
#include <vector>

#include "HelpFunctions.h"
#include "base/Random.hpp"

class RayTracer
{
public:
	static RayTracer& get()
    {
		static RayTracer* gpSingleton = nullptr;
		if (gpSingleton == nullptr)
		{
				gpSingleton = new RayTracer();
		}
		FW_ASSERT(gpSingleton != nullptr && "Failed to create RayTracer");
		return *gpSingleton;
    }


	void		startUp() { m_random = FW::Random(); }
	void		shutDown() { delete &get(); }

	Node*		constructHierarchy		(const std::vector<Triangle>&, std::vector<size_t>&);
	void		demolishTree			(Node*);	
	bool		rayCast					(const FW::Vec3f&, const FW::Vec3f&, Hit&, const std::vector<Triangle>&, const std::vector<size_t>&, Node*);

private:
	void		constructTree(size_t, size_t, Node*, Node*, std::vector<size_t>&);
	void		quickSort(int, int, Axis, std::vector<size_t>&);
	bool		isLeaf(Node*);
	void		createIndexList(std::vector<size_t>&, size_t);
	void		createBBForTriangles(const std::vector<Triangle>&, const std::vector<size_t>&);
	FW::Vec3f	getTriangleBBMaxPoint(size_t);
	FW::Vec3f	getTriangleBBMinPoint(size_t);
	FW::Vec3f	getTriangleBBCenPoint(size_t);

	RayTracer() {}
	~RayTracer() {}

	std::vector<FW::Vec3f> m_triangleBB;
	FW::Random	m_random;
};