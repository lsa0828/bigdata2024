# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:05:57 2024

@author: 이수아
"""

import numpy as np
import pandas as pd

[질의 3-1] emp.csv를 읽어서 DataFrame emp 만들기
df=pd.read_csv('emp.csv')

[질의 3-2] SELECT * FROM Emp;
df

[질의 3-3] SELECT ename FROM Emp;
df['ENAME']
df.ENAME
df.loc[:,'ENAME']
df.iloc[:,1]

[질의 3-4] SELECT ename, sal FROM Emp;
df[['ENAME', 'SAL']]
df.loc[:,['ENAME','SAL']]

[질의 3-5] SELECT DISTINCT job FROM Emp;
df.JOB.unique()

[질의 3-6] SELECT * FROM Emp WHERE sal < 2000;
cond=df.SAL < 2000
df[cond]
#df[df.SAL < 2000]

[질의 3-7] SELECT * FROM Emp WHERE sal BETWEEN 1000 AND 2000;
df[(df.SAL >= 1000)&(df.SAL <= 2000)]

[질의 3-8] SELECT * FROM Emp WHERE sal >= 1500 AND job= 'SALESMAN';
df[(df.SAL >= 1500)&(df.JOB == 'SALESMAN')]

[질의 3-9] SELECT * FROM Emp WHERE job IN ('MANAGER', 'CLERK');
df[df.JOB.isin(['MANAGER', 'CLERK'])]

[질의 3-10] SELECT * FROM Emp WHERE job NOT IN ('MANAGER', 'CLERK');
df[(df.JOB != 'MANAGER')&(df.JOB != 'CLERK')]

[질의 3-11] SELECT ename, job FROM Emp WHERE ename LIKE 'BLAKE';
df[['ENAME', 'JOB']][df.ENAME == 'BLAKE']

[질의 3-12] SELECT ename, job FROM Emp WHERE name LIKE '%AR%';
df[['ENAME', 'JOB']][df.ENAME.str.contains('AR')]

[질의 3-13] SELECT * FROM Emp WHERE ename LIKE '%AR%' AND sal >= 2000;
df[(df.ENAME.str.contains('AR')) & (df.SAL >= 2000)]

[질의 3-14] SELECT * FROM Emp ORDER BY ename;
df.sort_values(by='ENAME')

[질의 3-15] SELECT SUM(sal) FROM Emp;
df.SAL.sum()

[질의 3-16] SELECT SUM(sal) FROM Emp WHERE job LIKE 'SALESMAN';
df[df.JOB == 'SALESMAN'].SAL.sum()

[질의 3-17] SELECT SUM(sal), AVG(sal), MIN(sal), MAX(sal) FROM Emp;
df.SAL.agg(['sum', 'mean', 'min', 'max'])

[질의 3-18] SELECT COUNT(*) FROM Emp;
len(df)

[질의 3-19] SELECT COUNT(*), SUM(sal) FROM Emp GROUP BY job;
pd.concat([df.groupby('JOB').size(), df.groupby('JOB').SAL.sum()], axis=1)

[질의 3-20] SELECT * FROM Emp WHERE comm IS NOT NULL;
df[pd.notnull(df.COMM)]

[질의 4-0] emp.csv를 읽어서 DataFrame emp 만들기
df1=pd.read_csv('emp.csv')

[질의 4-1] emp에 age 열을 만들어 다음을 입력하여라(14명) 
[30,40,50,30,40,50,30,40,50,30,40,50,30,40]
df1=pd.concat([df1, pd.DataFrame([30,40,50,30,40,50,30,40,50,30,40,50,30,40], columns=['age'])], axis=1)

[질의 4-2] INSERT INTO Emp(empno, name, job) Values (9999, 'ALLEN', 'SALESMAN')
df1=pd.concat([df1, pd.DataFrame([[9999, 'ALLEN', 'SALESMAN']], columns=['EMPNO', 'ENAME', 'JOB'])])

[질의 4-3] emp의 ename='ALLEN' 행을 삭제하여라
(DELETE FROM emp WHERE ename LIKE 'ALLEN';)
df1=df1[df1.ENAME != 'ALLEN']

[질의 4-4] emp의 hiredate 열을 삭제하여라
(ALTER TABLE emp DROP COLUMN hiredate;)
df1.drop('HIREDATE', axis=1, inplace=True)

[질의 4-5] emp의 ename='SCOTT'의 sal을 3000으로 변경하여라
(UPDATE emp SET sal=3000 WHERE ename LIKE 'SCOTT';
df1.ENAME.isin(['SCOTT']).SAL=3000

[질의 5-1] emp의 sal 컬럼을 oldsal 이름으로 변경하여라. 
(ALTER TABLE emp RENAME sal TO oldsal;)
df1=df1.rename(columns={'SAL': 'OLDSAL'})

[질의 5-2] emp에 newsal 컬럼을 추가하여라, 값은 oldsal 컬럼값
(ALTER TABLE emp ADD newsal …;)
df1=pd.concat([df1, pd.DataFrame(list(df1.OLDSAL), columns=['NEWSAL'])], axis=1)

[질의 5-3] emp의 oldsal 컬럼을 삭제하여라
(ALTER TABLE emp DROP COLUMN oldsal;)
df1=df1.drop('OLDSAL', axis=1)