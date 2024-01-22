! Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
!
! All rights reserved.
!
! This file is provided to you to complete an assessment and for
! subsequent private study. It may not be shared and, in particular,
! may not be posted on the internet. Sharing this or any modified
! version may constitute academic misconduct under the University's
! regulations.

module io
  use iso_fortran_env, only: real64
  implicit none

  interface
     ! int read_column_data(char const* fn, int* nread, double** data)
     integer(c_int) function read_column_data_c(fn, nread, data) bind(C, name="read_column_data")
       use iso_c_binding
       character(len=1, kind=C_CHAR), intent(in) :: fn(*)
       integer(kind=c_int), intent(out) :: nread
       type(c_ptr), intent(out) :: data
     end function read_column_data_c

     ! void free_column_data(double** data)
     subroutine free_column_data_c(ptr) bind(C, name="free_column_data")
       use iso_c_binding
       type(c_ptr), intent(in) :: ptr
     end subroutine free_column_data_c

     ! int write_hist(char const* fn, int const nbins, double const* edges, int const* counts);
     integer(c_int) function write_hist_c(fn, nbins, edges, counts) bind(C, name="write_hist")
       use iso_c_binding
       character(len=1, kind=C_CHAR), intent(in) :: fn(*)
       integer(kind=c_int), value, intent(in) :: nbins
       real(c_double), intent(in) :: edges
       integer(c_int), intent(in) :: counts
     end function write_hist_c
  end interface

contains
  integer function read_column_data(fn, ans)
    use iso_c_binding

    character(len=*), intent(in) :: fn
    real(kind=real64), dimension(:), allocatable, intent(out) :: ans
    real(kind=real64), pointer :: data(:)

    character(len=1, kind=C_CHAR) :: c_fn(len_trim(fn) + 1)
    integer :: i, n
    integer(kind=c_int) :: ndata
    type(c_ptr) :: c_data

    ! Converting Fortran string to C string
    n = len_trim(fn)
    do i = 1, n
       c_fn(i) = fn(i:i)
    end do
    c_fn(n + 1) = C_NULL_CHAR

    read_column_data = read_column_data_c(c_fn, ndata, c_data)
    if (read_column_data == 0) then
       call c_f_pointer(c_data, data, (/ ndata /))
       allocate(ans(ndata))
       ans(:) = data(:)
       call free_column_data_c(c_data)
    end if
  end function read_column_data

  ! Write histogram data to the specified file.
  integer function write_hist(fn, edges, counts)
    use iso_c_binding

    character(len=*), intent(in) :: fn
    real(kind=real64), dimension(:), intent(in) :: edges
    integer, dimension(:), intent(in) :: counts

    character(len=1, kind=C_CHAR) :: c_fn(len_trim(fn) + 1)
    integer :: i, n
    integer(kind=c_int) :: nbins

    ! Converting Fortran string to C string
    n = len_trim(fn)
    do i = 1, n
       c_fn(i) = fn(i:i)
    end do
    c_fn(n + 1) = C_NULL_CHAR

    nbins = size(counts)
    write_hist = write_hist_c(c_fn, nbins, edges(1), counts(1))
  end function write_hist
end module io
