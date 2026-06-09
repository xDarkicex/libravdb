//go:build !windows

package singlefile

import "syscall"

const oNoFollow = syscall.O_NOFOLLOW
