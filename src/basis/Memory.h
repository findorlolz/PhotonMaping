#pragma once

#include <cstdlib>
#include <iostream>
#include <assert.h>

class Actor;

#include <omp.h>
 
/** This is a class for guard objects using OpenMP
 *  It is adapted from the book
 *  "Pattern-Oriented Software Architecture". */
class omp_guard {
public:
    omp_guard (omp_lock_t &lock);
    void acquire ();
    void release ();
    ~omp_guard ();
 
private:
    omp_lock_t *lock_;
    bool owner_;
   
    omp_guard (const omp_guard &);
    void operator= (const omp_guard &);
};

class MemPool
{
public:
	MemPool() {}

	void MemPool::startUp(const size_t sizeOfBlock, const size_t numberOfBlocks)
	{
		size_ = numberOfBlocks;
		unsigned char* ptr = reinterpret_cast<unsigned char*>(malloc(sizeOfBlock*numberOfBlocks));
		begin_ = ptr;
		cursor_ = ptr;

		size_t nextAddress;
		size_t currentAddress = reinterpret_cast<size_t>(ptr);
		for(auto i = 0u; i < numberOfBlocks - 1; ++i)
		{
			nextAddress = currentAddress + sizeOfBlock;
			unsigned char** tmp = reinterpret_cast<unsigned char**>(currentAddress);
			tmp[0] = reinterpret_cast<unsigned char*>(nextAddress);
			currentAddress = nextAddress;
		}

		lastMemberOfPool_ = reinterpret_cast<unsigned char*>(currentAddress);
		unsigned char** tmp = reinterpret_cast<unsigned char**>(currentAddress);
		tmp[0] = nullptr;
		lock_ = omp_lock_t();
		omp_init_lock (&lock_);

	}

	void MemPool::shutDown()
	{	
		free(begin_);
		begin_= nullptr;
		cursor_ = nullptr;
	}

	void* MemPool::alloc()
	{
		void* v = nullptr;
		{
			omp_guard my_guard(lock_);
			size_--;
			v = (void*) cursor_;
			unsigned char** tmp = reinterpret_cast<unsigned char**>(cursor_);
			cursor_ = tmp[0];
		}
		return v;
	}


	void MemPool::release(unsigned char* ptr)
	{
		{
			omp_guard my_guard(lock_);
			if(cursor_ == nullptr)
				cursor_ = ptr;

			size_++;
			unsigned char** tmp = reinterpret_cast<unsigned char**>(lastMemberOfPool_);
			tmp[0] = ptr;
			tmp = reinterpret_cast<unsigned char**>(ptr);
			tmp[0] = nullptr;
			lastMemberOfPool_ = ptr;
		}
	}

private:

	unsigned char* lastMemberOfPool_;
	unsigned char* begin_;
	unsigned char* cursor_;
	size_t size_;
	omp_lock_t lock_;
};