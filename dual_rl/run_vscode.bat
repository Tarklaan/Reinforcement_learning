set CONDAPATH=%CONDAPATH%
set ENVNAME=RL_Trading

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

cd /d "%~dp0"
code .

