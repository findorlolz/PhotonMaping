#define _CRT_SECURE_NO_WARNINGS

#include "base/Defs.hpp"
#include "base/Math.hpp"
#include "RayTracer.h"
#include <stdio.h>
#include "rtIntersect.inl"
#include <iostream>

Node* RayTracer::constructHierarchy(const std::vector<Triangle>& triangles, std::vector<size_t>& indexList)
{
	std::cout << "Creating RayTracer Hieracrhy... ";
	size_t size = triangles.size();
	Node* root = new Node();	
	
	createIndexList(indexList, size);
	createBBForTriangles(triangles, indexList);
	constructTree(0, size-1 , root, root, indexList);
	
	m_triangleBB.clear();

	std::cout << "done!" << std::endl;
	return root;
}

void RayTracer::constructTree(size_t startPrim, size_t endPrim, Node* node, Node* root,std::vector<size_t>& indexList)
{
	node->startPrim = startPrim;
	node->endPrim = endPrim;
	
	node->leftChild = NULL;
	node->rightChild = NULL;
	Axis axis;

	if (node == root)
	{
		node->BBMin = FW::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
		node->BBMax = FW::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		
		for (int prim = startPrim; prim <= endPrim; prim++)
		{
			node->BBMin = min(node->BBMin, RayTracer::getTriangleBBMinPoint(indexList[prim]));
			node->BBMax = max(node->BBMax, RayTracer::getTriangleBBMaxPoint(indexList[prim]));
		}
	}

	if ((endPrim - startPrim) + 1 > 8)
	{
		const size_t bbSize = (endPrim - startPrim + 1);
		const size_t index = bbSize - 1;
		float tmpSAH = FLT_MAX;
		size_t tmpSAHAxis = 0;
		size_t tmpSAHindex = 0;

		Node* leftNode = new Node();
		Node* rightNode = new Node();

		std::vector<FW::Vec3i> sortedIndices = std::vector<FW::Vec3i>(bbSize);
		std::vector<BB> boundingBoxesR = std::vector<BB>(index);
		std::vector<BB> boundingBoxesL =  std::vector<BB>(index);

		for(auto i = 0u; i < 3; ++i)
		{
			quickSort((int)startPrim, (int)endPrim, (Axis) i, indexList);
			FW::Vec3f maxPointRight = FW::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			FW::Vec3f maxPointLeft = FW::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			FW::Vec3f minPointRight = FW::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
			FW::Vec3f minPointLeft = FW::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
			for(auto j = 0u; j < index; ++j)
			{
				sortedIndices[j][i] = indexList[startPrim+j];
				minPointLeft = min(minPointLeft, getTriangleBBMinPoint(indexList[startPrim+j]));
				maxPointLeft = max(maxPointLeft, getTriangleBBMaxPoint(indexList[startPrim+j]));
				boundingBoxesL[j].set(minPointLeft, maxPointLeft);
				minPointRight = min(minPointRight, RayTracer::getTriangleBBMinPoint(indexList[endPrim-j]));
				maxPointRight = max(maxPointRight, RayTracer::getTriangleBBMaxPoint(indexList[endPrim-j]));
				boundingBoxesR[index-(j+1)].set(minPointRight, maxPointRight);
			}
			sortedIndices[index][i] = indexList[startPrim+index];
			for(auto j = 0u; j < index; ++j)
			{
				float SAH = boundingBoxesL[j].area * (float)(j+1)  + boundingBoxesR[j].area * (float)(index - j);
				if(SAH < tmpSAH)
				{
					leftNode->BBMin = boundingBoxesL[j].min;
					leftNode->BBMax = boundingBoxesL[j].max;
					rightNode->BBMin = boundingBoxesR[j].min;
					rightNode->BBMax = boundingBoxesR[j].max;
					tmpSAH = SAH;
					tmpSAHAxis = i;
					tmpSAHindex = j;
				}
			}
		}

		std::cout << "SAH: " << tmpSAH << " index " << tmpSAHindex << " Axis: " << tmpSAHAxis << std::endl;
		node->axis = (Axis)tmpSAHAxis;
		node->leftChild = leftNode;
		node->rightChild = rightNode;

		size_t t = 0;
		for (auto i = startPrim; i <= endPrim; i++)
		{
			indexList[i] = sortedIndices[t][tmpSAHAxis]; 
			t++;
		}

		size_t split = startPrim + tmpSAHindex;
			
		constructTree(startPrim, split, leftNode, root, indexList);
		constructTree(split + 1, endPrim, rightNode, root, indexList);
	}
}

void RayTracer::quickSort(int start, int end, Axis axis, std::vector<size_t>& indexList)
{
    if (start > end)
            return;

    // Randomize partition
    int rndm = m_random.getU32(start, end);
	std::swap(indexList[rndm], indexList[end]);

    // Partition
	float pivot = RayTracer::getTriangleBBCenPoint(indexList[end])[axis];
    int i = start - 1;
    for (int j = start; j < end; j++)
    {
            if (getTriangleBBCenPoint(indexList[j])[axis] < pivot)
            {
                    i++;
                    std::swap(indexList[i], indexList[j]);
            }
    }
    std::swap(indexList[i+1], indexList[end]);
    int middle = i+1;

    // Recursively sort partitions
    quickSort(start, middle-1, axis, indexList);
    quickSort(middle+1, end, axis, indexList);
}

void RayTracer::demolishTree(Node* node)
{
	
	if(!isLeaf(node))
	{
		RayTracer::demolishTree(node->leftChild);
		RayTracer::demolishTree(node->rightChild);
	}
	delete node;

}

bool RayTracer::rayCast(const FW::Vec3f& orig, const FW::Vec3f& dir, Hit& closestHit, const std::vector<Triangle>& triangles, const std::vector<size_t>& indexList, Node* root)
{
	Node* current = root;
	
	//Basic recursion traverse test
	//RayTracer::basicTraverseTest(orig, dir, closestHit, current);

	FW::Vec3f dirInv = 1.0f/(dir);
	Node* stack[128];
	size_t stackPointer = 0;

	while (true)
	{
		bool intersection = intersect_bb(&orig.x, &dirInv.x, &(current->BBMin.x), &(current->BBMax.x), closestHit.t);	
		if (intersection)
		{
			if(isLeaf(current))
			{
				for (size_t i = current->startPrim; i <= current->endPrim; ++i)
				{
					float t, u, v;
					if ( intersect_triangle1( &orig.x, &dir.x,
											  &triangles[indexList[i]].m_vertices[0]->x,
											  &triangles[indexList[i]].m_vertices[1]->x,
											  &triangles[indexList[i]].m_vertices[2]->x,
											  t, u, v ) )
					{
						if ( t > 0.0f && t < closestHit.t )
						{
							closestHit.i = indexList[i];
							closestHit.t = t;
							closestHit.u = u;
							closestHit.v = v;
							closestHit.b = true;
						}
					}
				}
				if (stackPointer == 0) // stack empty, break
				{
					break;
				}
				else // stack not empty, pop new current
				{
					stackPointer -=1;
					current = stack[stackPointer];
				}
			}
			else // isn't leaf
			{
				//std::cout << "Ray direction: " << dir[current->axis] << std::endl;
				if (dir[current->axis] < 0.0f)
				{
					stack[stackPointer] = current->leftChild;
					stackPointer = stackPointer + 1;
					current = current->rightChild;
					//std::cout << "Go left" << std::endl;
				}
				else
				{
					stack[stackPointer] = current->rightChild;
					stackPointer = stackPointer + 1;
					current = current->leftChild;
					//std::cout<< "Go right" << std::endl;
				}
			}		//isLeaf-if
		}
		else // doesnt intersect with current node's bb
		{
			if (stackPointer == 0)
			{
				break;
			}
			else
			{
				stackPointer -= 1;
				current = stack[stackPointer];
				//std::cout << "stack pop 1 " << std::endl;
			}
		}			//intesection if
	}				//while-loop
	
	//std::cout << "end" <<std::endl;
	if(closestHit.b)
	{
		closestHit.intersectionPoint = orig + closestHit.t*dir;
		closestHit.triangle = triangles[closestHit.i];
		return true;
	}
	else 
		return false;
}

bool RayTracer::isLeaf(Node* node)
{
	if (node->leftChild == NULL)
		return true;
	else
		return false;
}

void RayTracer::createIndexList(std::vector<size_t>& list, size_t size)
{
	list = std::vector<size_t> (size);
	for (int i = 0; i < size; i++)
	{
		list[i] = i;
	}
}

void RayTracer::createBBForTriangles(const std::vector<Triangle>& triangles, const std::vector<size_t>& indexList)
{
	m_triangleBB = std::vector<FW::Vec3f>(3*triangles.size());
	size_t i = 0u;
	while (i < triangles.size())
	{

		m_triangleBB[3*i] = FW::Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
		m_triangleBB[3*i + 1] = FW::Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		for(auto vert = 0u; vert < 3; vert++)
		{
			FW::Vec3f tmp = *(triangles[indexList[i]].m_vertices[vert]);
			m_triangleBB[3*i] = min(tmp, m_triangleBB[3*i]);
			m_triangleBB[3*i + 1] = max(tmp, m_triangleBB[3*i + 1]);
		}
		m_triangleBB[3*i + 2] = (m_triangleBB[3*i] + m_triangleBB[3*i+1])*0.5f;
		i++;
	}
}

FW::Vec3f RayTracer::getTriangleBBMinPoint(size_t index)
{
	return m_triangleBB[3*index];
}

FW::Vec3f RayTracer::getTriangleBBMaxPoint(size_t index)
{
	return m_triangleBB[3*index + 1];
}

FW::Vec3f RayTracer::getTriangleBBCenPoint(size_t index)
{
	return m_triangleBB[3*index + 2];
}