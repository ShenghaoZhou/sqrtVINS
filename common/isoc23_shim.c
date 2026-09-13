/* Link-time shim: Ocean static archives reference the glibc 2.38+ symbols
   (__isoc23_*) which are absent from the conda linker sysroot. At runtime the
   system glibc provides them; these aliases only satisfy the static link. */
#include <stdlib.h>
long __isoc23_strtol(const char *n, char **e, int b) { return strtol(n, e, b); }
unsigned long __isoc23_strtoul(const char *n, char **e, int b) { return strtoul(n, e, b); }
long long __isoc23_strtoll(const char *n, char **e, int b) { return strtoll(n, e, b); }
unsigned long long __isoc23_strtoull(const char *n, char **e, int b) { return strtoull(n, e, b); }
