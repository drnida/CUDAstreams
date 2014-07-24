#include <stdio.h>
#include <stdlib.h>

int main() {
   char *string;
   int size = 2000;
   time_t t;
   int num;
   int i;
   string = (char*)malloc(sizeof(char)*size);
   srand((unsigned)time(&t));
   for(i = 0; i < size - 1; i++) {
      num = rand() % 57;
      num += 'A';
      string[i] = num; 
   }
   string[4] = 'h';
   string[5] = 'e';
   string[6] = 'l';
   string[7] = 'l';
   string[8] = 'o';

   string[554] = 'h';
   string[555] = 'e';
   string[556] = 'l';
   string[557] = 'l';
   string[558] = 'o';

   string[1004] = 'h';
   string[1005] = 'e';
   string[1006] = 'l';
   string[1007] = 'l';
   string[1008] = 'o';
   string[1022] = 'b';
   string[size - 1] = '\0';

   printf("%s", string);
}
