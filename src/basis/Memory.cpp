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