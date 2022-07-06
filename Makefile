CFLAGS = -O2 -Wall -Wextra -Wpedantic
OBJ = src/libwang.so
SRC = src/libwang.c

all: src/libwang.so src/tags

src/libwang.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

src/tags: $(SRC)
	ctags $(SRC)
	mv tags src/

clean:
	rm src/tags $(OBJ)
