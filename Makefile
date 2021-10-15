#/bin/bash

OUTNAME := $(notdir $(shell pwd))

CC ?= gcc
CPPFLAGS += -Wall -O3

LDFLAGS = -O3 
LDLIBS = -lOpenCL -lm

objects = main.o common.o
	
$(OUTNAME).elf : $(objects)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

matmul.o : main.c defs.h 
common.o : common.c

.PHONY : clean 
clean : 
	-rm -f *.elf *.o 
