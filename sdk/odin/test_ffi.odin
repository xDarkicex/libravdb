package main

import "core:fmt"

when ODIN_OS == .Darwin {
    foreign import libravdb "../cgo/libravdb.dylib"
} else when ODIN_OS == .Linux {
    foreign import libravdb "../cgo/libravdb.so"
}

@(default_calling_convention="c")
foreign libravdb {
    OpenDB :: proc(path: cstring) -> i32 ---
}

main :: proc() {
    id := OpenDB("hello")
    fmt.println("DB ID:", id)
}
