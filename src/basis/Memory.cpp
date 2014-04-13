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
	m_size = size;
	m_nextIndex = 1;
	m_array = ptr;
	(*m_array).reserve(m_size+1);
	HeapNode tmp;
	tmp.key = -1337;
	tmp.value = 1337;
	(*m_array).push_back(tmp);
}

MaxHeap::~MaxHeap()
{
}

float MaxHeap::getMaxKey()
{
	return (*m_array)[1].key;
}

size_t MaxHeap::getMaxValue()
{
	return (*m_array)[1].value;
}

bool MaxHeap::heapIsFull()
{
	if (m_nextIndex > m_size)
		return true;
	else
		return false;
}

bool MaxHeap::isLeaf(size_t i)
{
	if (getLeftChild(i) > m_nextIndex)
		return true;
	else
		return false;
}

void MaxHeap::balanceTreeRec(size_t i)
{
	size_t l = getLeftChild(i);
	size_t r = getRightChild(i);
	size_t greatest;

	if (l < m_nextIndex && (*m_array)[l].key > (*m_array)[i].key)
		greatest = l;
	else
		greatest = i;

	if (r < m_nextIndex && (*m_array)[r].key > (*m_array)[greatest].key)
		greatest = r;

	if (greatest != i)
	{
		swapNode(i, greatest);
		balanceTreeRec(greatest);
	}
}

bool MaxHeap::pushHeap(size_t value, float key)
{
	if (heapIsFull())
	{
		if(key > getMaxKey())
			return false;
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		swapNode(1,m_nextIndex-1);
		(*m_array)[m_nextIndex-1] = node;
		balanceTreeRec(1);
		return true;
	}
	else if (m_nextIndex == m_size)
	{
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		(*m_array).push_back(node);
		m_nextIndex++;
		balanceEntireTree();
		return true;
	}
	else
	{
		HeapNode node = HeapNode();
		node.key = key;
		node.value = value;
		(*m_array).push_back(node);
		m_nextIndex++;
		return true;
	}
}

void MaxHeap::swapNode(size_t i, size_t j)
{
	HeapNode tmp = (*m_array)[i];
	(*m_array)[i] = (*m_array)[j];
	(*m_array)[j] = tmp;
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