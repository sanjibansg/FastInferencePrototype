#include </usr/include/python3.8/Python.h>

int main(){
printf("Getting Python information\n");
Py_Initialize();
if( !Py_IsInitialized() )
{
puts("Unable to initialize Python interpreter.");
return 1;
}
printf("Prefix: %s\nExec Prefix: %s\nPython Path %s\n",(char*)Py_GetPrefix(),(char*)Py_GetExecPrefix(),(char*)Py_GetProgramFullPath());
printf("Module Path: %s\n",(char*)Py_GetPath());
printf("Version: %s\nPlatform: %s\nCopyright: %s\n",Py_GetVersion(),Py_GetPlatform(),Py_GetCopyright());
printf("Compiler String: %s\nBuild Info: %s\n",Py_GetCompiler(),Py_GetBuildInfo());
Py_Finalize();
return 0;
}
