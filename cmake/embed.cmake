# Wraps a text file into a C++ raw string literal: cmake -DIN=... -DOUT=... -DNAME=... -P embed.cmake
file(READ "${IN}" body)
file(WRITE "${OUT}" "static const char ${NAME}[] = R\"KVB_SRC(${body})KVB_SRC\";\n")
