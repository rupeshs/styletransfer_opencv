#Author :Rupesh Sreeraman
#This file is a part of rlab

CC =g++
SOURCES =main.cpp
CFLAGS=-g -I ../include  
LDFLAGS=-L ../lib -lopencv_core400.dll -lopencv_highgui400 -lopencv_imgproc400 -lopencv_videoio400 -lopencv_dnn400.dll -lopencv_objdetect400.dll -lopencv_video400.dll -lopencv_calib3d400.dll -lopencv_imgcodecs400.dll -lopencv_video400.dll
BINDIR=../bin
EXECUTABLE =$(BINDIR)/stylecv.exe

OBJECTS	:= $(SOURCES:.cpp=.o)

all:$(OBJECTS)
	$(CC) -o $(EXECUTABLE) $(OBJECTS) $(LDFLAGS)  

 %.o: %.cpp
	$(CC) $(CFLAGS) -o $(OBJECTS) -c $(SOURCES) 

clean:
	rm -f $(EXECUTABLE) *.o

rebuild: clean all

