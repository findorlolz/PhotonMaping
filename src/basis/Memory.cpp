#include "Memory.h"

omp_guard::omp_guard (omp_lock_t &lock) : lock_ (&lock)
    , owner_ (false)
{
    acquire ();
}
 
void omp_guard::acquire ()
{
    omp_set_lock (lock_);
    owner_ = true;
}
 
void omp_guard::release ()
{
    if (owner_) {
        owner_ = false;
        omp_unset_lock (lock_);
    }
}
 
omp_guard::~omp_guard ()
{
    release ();
}

MaxHeap::MaxHeap(size_t size, std::vector<HeapNode>* ptr)
{
	m_size = size+1;
	m_heapCounter = 1;
	m_array = ptr;
	(*m_array).reserve(m_size);
	(*m_array).push_back(HeapNode());
}

MaxHeap::~MaxHeap()
{
}

size_t MaxHeap::getMaxKey()
{
	return (*m_array)[1].key;
}

float MaxHeap::getMaxValue()
{
	return (*m_array)[1].value;
}

bool MaxHeap::heapIsFull()
{
	if (m_heapCounter == m_size)
		return true;
	else
		return false;
}

bool MaxHeap::isLeaf(size_t i)
{
	if (getLeftChild(i) > m_heapCounter)
		return true;
	else
		return false;
}

void MaxHeap::balanceTreeRec(size_t i)
{
	size_t l = getLeftChild(i);
	size_t r = getRightChild(i);
	size_t greatest;

	if (l < m_heapCounter && (*m_array)[l].value > (*m_array)[i].value)
		greatest = l;
	else
		greatest = i;

	if (r < m_heapCounter && (*m_array)[l].value < (*m_array)[greatest].value)
		greatest = r;

	if (greatest != i)
	{
		swapNode(i, greatest);
		balanceTreeRec(greatest);
	}
}

void MaxHeap::pushHeap(size_t key, float value)
{
	if (heapIsFull())
	{
		if(value > getMaxValue())
			return;
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		(*m_array)[1] = node;
		balanceTreeRec(1);
	}
	else if (m_heapCounter == m_size - 1)
	{
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		(*m_array).push_back(node);
		m_heapCounter++;
		balanceEntireTree();
	}
	else
	{
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		(*m_array).push_back(node);
		m_heapCounter++;
	}
}

void MaxHeap::swapNode(size_t i, size_t j)
{
	(*m_array)[0] = (*m_array)[i];
	(*m_array)[i] = (*m_array)[j];
	(*m_array)[j] = (*m_array)[0];
}

void MaxHeap::balanceEntireTree()
{
	MaxHeap::balanceEntireTreeRec(1);
}

void MaxHeap::balanceEntireTreeRec(size_t i)
{
	if (isLeaf(i))
		return;
	balanceEntireTreeRec(getLeftChild(i));
	balanceEntireTreeRec(getRightChild(i));
	balanceTreeRec(i);
}