! Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
!
! All rights reserved.
!
! This file is provided to you to complete an assessment and for
! subsequent private study. It may not be shared and, in particular,
! may not be posted on the internet. Sharing this or any modified
! version may constitute academic misconduct under the University's
! regulations.

module hist_cpu
  use iso_fortran_env, only: real64
  implicit none

contains
  ! Compute the counts of data in the bins.
  !
  ! If nbins is the number of bins, then:
  !
  ! bin_edges - the bin edges. It must have size nbins+1, be sorted,
  ! and have no zero-size bins, i.e.:
  !
  !   bin_edges(i) < bin_edges(i+1)
  !
  ! data - the data to be histogrammed
  !
  ! dt - time spent on work (seconds) for benchmarking
  !
  ! counts - output parameter. Must have size nbins. Element i holds
  ! the count of elements in data >= bin_edges(i) and < bin_edges(i+1)
  !
  subroutine compute_histogram_cpu(bin_edges, data, dt, counts)
    real(kind=real64), dimension(:), intent(in) :: bin_edges
    real(kind=real64), dimension(:), intent(in) :: data
    integer, dimension(:), intent(out) :: counts
    real(kind=real64), intent(out) :: dt

    integer :: nedges, nbins, ncounts, ndata
    integer :: i
    integer :: ub
    real(kind=real64) :: tstart, tend

    nedges = size(bin_edges)
    nbins = nedges - 1
    ncounts = size(counts)
    ndata = size(data)

    ! Sanity checks
    if (nbins .ne. ncounts) then
       print *, "Size mismatch on edges and counts"
       return
    end if
    
    if (nbins <= 0) then
       print *, "Require at least one bin, have ", nbins
       return
    end if

    do i = 1, nbins
       if (bin_edges(i) >= bin_edges(i+1)) then
          print *,  "Bin edges not increasing at index ", i
          return
       end if
    end do

    ! Work begins
    call cpu_time(tstart)

    ! Zero result array
    counts = 0
    ! Main calculation
    do i = 1, ndata
       ub = upper_bound(bin_edges, data(i))
       if (ub == 1 ) then
          ! value below all bins
       else if (ub == nedges + 1) then
          ! value above all bins
       else
          ! in a bin!
          counts(ub - 1) = counts(ub - 1) + 1
       end if
    end do
    call cpu_time(tend)
    dt = tend - tstart
  end subroutine

  ! Return index of first element in range greater than value, or len if
  ! not found.
  integer pure function upper_bound(data, val)
    real(kind=real64), dimension(:), intent(in) :: data
    real(kind=real64), intent(in) :: val

    integer :: begin
    integer :: len
    integer :: half
    integer :: mid

    begin = 1
    len = size(data)
    do while (len > 0)
       half = len / 2
       mid = begin + half
       if ( val < data(mid) ) then
          len = half
       else
          begin = mid + 1
          len = len - half - 1;
       end if
    end do

    upper_bound = begin
  end function upper_bound

end module hist_cpu
