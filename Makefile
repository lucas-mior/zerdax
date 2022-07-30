CFLAGS = -O2 -Wall -Wextra -Wpedantic
OBJ = src/libffilter.so
SRC = src/libffilter.c

all: src/libffilter.so src/tags

src/libffilter.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

src/tags: $(SRC)
	ctags $(SRC)
	mv tags src/

clean:
	rm src/tags $(OBJ)
