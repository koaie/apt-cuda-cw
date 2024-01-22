! Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
!
! All rights reserved.
!
! This file is provided to you to complete an assessment and for
! subsequent private study. It may not be shared and, in particular,
! may not be posted on the internet. Sharing this or any modified
! version may constitute academic misconduct under the University's
! regulations.

#define TEST_CASE(name) subroutine TEST_CASE_ ## name (test_err, test_nass); \
  implicit none; \
  logical :: test_err; \
  integer :: test_nass; \

#define END_TEST_CASE end subroutine

#define TEST_INIT() \
  logical :: test_err; \
  integer :: test_nass;

#define RUN_TEST(name) call test_runner( #name, TEST_CASE_ ## name)
#define REQUIRE(expr) \
  test_nass = test_nass + 1; \
  if (expr) then; \
  else ;\
     print *, "  line ", __LINE__, ": ", #expr; \
     test_err = .true. ;\
     return ;\
  end if
#define REQUIRE_APPROX(actual, expected, margin)	\
  REQUIRE(abs(actual - expected) .lt. margin)

module tester
  implicit none
  interface
     subroutine test_ptr(err, n)
       logical, intent(inout) :: err
       integer, intent(inout) :: n
     end subroutine
  end interface
contains
  subroutine test_runner(name, test_sub)
    character(len=*) :: name
    procedure(test_ptr) :: test_sub

    logical :: err
    integer :: n_ass

    err = .false.
    n_ass = 0

    call test_sub(err, n_ass)

    if (err) then
       ! Error
       write (*,"(a20,': FAIL')") name
    else
       ! OK
       write(*, "(a20,': PASS with ',i2,' assertions')") name, n_ass
    end if
  end subroutine
end module

program test
  use iso_fortran_env, only: real64

  use tester
  use hist_cpu
  use io

  implicit none

  RUN_TEST(upper_bound)
  RUN_TEST(hist)
  ! RUN_TEST(timing)
  RUN_TEST(read_data)
contains

  TEST_CASE(upper_bound)
    real(kind=real64), dimension(6) :: edges
    real(kind=real64), dimension(6) :: data
    integer, dimension(6) :: expected
    integer :: ub
    integer :: i

    edges = (/ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 /)
    data = (/ 0.5, 1.5, 1.0, 4.5, 5.0, 6.0 /)
    expected = (/ 2, 3, 3, 6, 7, 7 /)

    do i = 1, size(data)
       ub = upper_bound(edges, data(i))
       REQUIRE(ub .eq. expected(i))
    end do
  END_TEST_CASE

  TEST_CASE(hist)
    real(kind=real64), dimension(6) :: edges
    real(kind=real64), dimension(8) :: data
    integer :: nedges, nbins, ndata, i
    integer, dimension(5) :: h
    integer, dimension(5) :: expected
    real(kind=real64) :: t

    edges = (/ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 /)
    nedges = size(edges)
    nbins = nedges - 1
    data = (/ -0.5, 0.0, 0.5, 1.5, 1.0, 4.5, 5.0, 6.0 /)
    ndata = size(data)

    REQUIRE(size(h) == nbins)
    call compute_histogram_cpu(edges, data, t, h)

    expected = (/ 2, 2, 0, 0, 1 /)
    REQUIRE(size(expected) == nbins)
    do i = 1, nbins
       REQUIRE(h(i) == expected(i));
    end do
  END_TEST_CASE

  TEST_CASE(read_data)
    real(kind=real64), dimension(:), allocatable :: data
    integer :: err
    err = read_column_data("../sample_data/uniform.bins", data)
    REQUIRE(err == 0)
    REQUIRE(size(data) == 11)
    err = read_column_data("../sample_data/uni_100k.dat", data)
    REQUIRE(err == 0)
    REQUIRE(size(data) == 100000)
    REQUIRE_APPROX(data(1), 6.31278952142d-1, 1d-12)
    REQUIRE_APPROX(data(size(data)), 3.03617733625d-1, 1d-12)
  END_TEST_CASE

end program test
