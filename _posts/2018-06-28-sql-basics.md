---
layout:         post
title:          SQL Basics
subtitle:
card-image:     /mldl/assets/images/cards/cat19.gif
date:           2018-06-28 09:00:00
tags:           [database]
categories:     [database]
post-card-type: image
mathjax:        true
---

* <a href="#Introduction to the Relational Model">Introduction to the Relational Model
* <a href="#Structure of Relational Databases">Structure of Relational Databases
* <a href="#Database Schema">Database Schema
* <a href="#Relational Operations">Relational Operations
* <a href="#SQL Data Definition">SQL Data Definition
* <a href="#Domain Types in SQL">Domain Types in SQL
* <a href="#Create Table Construct">Create Table Construct
* <a href="#Drop and Alter Table Constructs">Drop and Alter Table Constructs
* <a href="#Basic Structure of SQL Queries">Basic Structure of SQL Queries
* <a href="#The select Clause">The select Clause
* <a href="#The where Clause">The where Clause
* <a href="#Queries on Multiple Relations">Queries on Multiple Relations
* <a href="#Natural Join">Natural Join
* <a href="#The Rename Operation">The Rename Operation
* <a href="#String Operations">String Operations
* <a href="#Ordering the Display of Tuples">Ordering the Display of Tuples
* <a href="#Where Clause Predicates">Where Clause Predicates
* <a href="#Set Operations">Set Operations
* <a href="#Null Values">Null Values
* <a href="#Aggregate Functions">Aggregate Functions
* <a href="#Nested Subqueries">Nested Subqueries
* <a href="#Modification of the Database">Modification of the Database


## <a name="Introduction to the Relational Model">Introduction to the Relational Model</a>

The relational model is today the primary data model for commercial data- processing applications. It attained its primary position because of its simplicity, which eases the job of the programmer, compared to earlier data models such as the network model or the hierarchical model.

## <a name="Structure of Relational Databases">Structure of Relational Databases</a>

A relational database consists of a collection of **tables**, each of which is assigned a unique name. In the relational model the term **relation** is used to refer to a table, while the term **tuple** is used to refer to a row. Similarly, the term **attribute** refers to a column of a table. We use the term **relation instance** to refer to a specific instance of a relation, i.e., containing a specific set of rows.

For each attribute of a relation, there is a set of permitted values, called the **domain** of that attribute. We require that, for all relations r, the domains of all attributes of r be atomic. A domain is **atomic** if elements of the domain are considered to be indivisible units. For example, suppose the table _instructor_ had an attribute _phone number_, which can store a set of phone numbers corresponding to the instructor. Then the domain of _phone number_ would not be atomic, since an element of the domain is a set of phone numbers, and it has subparts, namely the individual phone numbers in the set. We assume that all attributes have atomic domains.

The **null** value is a special value that signifies that the value is unknown or does not exist. We shall see later that null values cause a number of difficulties when we access or update the database, and thus should be eliminated if at all possible.

## <a name="Database Schema">Database Schema</a>

When we talk about a database, we must differentiate between the **database schema**, which is the logical design of the database, and the **database instance**, which is a snapshot of the data in the database at a given instant in time.

The concept of a relation corresponds to the programming-language notion of a variable, while the concept of a **relation schema** corresponds to the programming-language notion of type definition. In general, a relation schema consists of a list of attributes and their corresponding domains.

The concept of a relation instance corresponds to the programming-language notion of a value of a variable. The value of a given variable may change with time; similarly the contents of a relation instance may change with time as the relation is updated. In contrast, the schema of a relation does not generally change.

Although it is important to know the difference between a relation schema and a relation instance, we often use the same name, such as instructor, to refer to both the schema and the instance. Where required, we explicitly refer to the schema or to the instance, for example “the _instructor_ schema,” or “an instance of the _instructor_ relation.” However, where it is clear whether we mean the schema or the instance, we simply use the relation name.

Consider the _department_ relation, the schema for that relation is: _department (dept name, building, budget)_.

^## Keys

We must have a way to specify how tuples within a given relation are distinguished. This is expressed in terms of their attributes. That is, the values of the attribute values of a tuple must be such that they can uniquely identify the tuple. In other words, no two tuples in a relation are allowed to have exactly the same value for all attributes.

A **superkey** is a set of one or more attributes that, taken collectively, allow us to identify uniquely a tuple in the relation.

A superkey may contain extraneous attributes. If K is a superkey, then so is any superset of K. We are often interested in superkeys for which no proper subset is a superkey. Such minimal superkeys are called **candidate keys**.

We shall use the term **primary key** to denote a candidate key that is chosen by the database designer as the principal means of identifying tuples within a relation. The primary key should be chosen such that its attribute values are never, or very rarely, changed. It is customary to list the primary key attributes of a relation schema before the other attributes.

A relation, say $$r_1$$, may include among its attributes the primary key of another relation, say $$r_2$$. This attribute is called a **foreign key** from $$r_1$$, referencing $$r_2$$. The relation $$r_1$$  is also called the referencing relation of the foreign key dependency, and $$r_2$$ is called the referenced relation of the foreign key.

^## Schema Diagrams

A database schema, along with primary key and foreign key dependencies, can be depicted by **schema diagrams**. The following figure shows the schema diagram for our university organization. Each relation appears as a box, with the relation name at the top in blue, and the attributes listed inside the box. Primary key attributes are shown underlined. Foreign key dependencies appear as arrows from the foreign key attributes of the referencing relation to the primary key of the referenced relation.

![sql1](/mldl/assets/images/sql1.jpg)

## <a name="Relational Operations">Relational Operations</a>

All procedural relational query languages provide a set of operations that can be applied to either a single relation or a pair of relations. These operations have the nice and desired property that their result is always a single relation. This property allows one to combine several of these operations in a modular way. Specifically, since the result of a relational query is itself a relation, relational operations can be applied to the results of queries as well as to the given set of relations.

The most frequent operation is the **selection of specific tuples** from a single relation (say _instructor_) that satisfies some particular predicate (say _salary_ > $85,000). The result is a new relation that is a subset of the original relation (_instructor_).

Another frequent operation is to **select certain attributes** (columns) from a relation. The result is a new relation having only those selected attributes.

The **join** operation allows the combining of two relations by merging pairs of tuples, one from each relation, into a single tuple. There are a number of different ways to join relations.

In the form of a **natural join**, a tuple from the _instructor_ relation matches a tuple in the _department_ relation if the values of their _dept name_ attributes are the same. All such matching pairs of tuples are present in the joint result. In general, the natural join operation on two relations matches tuples whose values are the same on all attribute names that are common to both relations.

The **Cartesian product** operation combines tuples from two relations, but unlike the join operation, its result contains all pairs of tuples from the two relations, regardless of whether their attribute values match.

Because relations are sets, we can perform normal set operations on relations. The **union** operation performs a set union of two “similarly structured” tables. Other set operations, such as **intersection** and **set difference** can be performed as well.

## <a name="SQL Data Definition">SQL Data Definition</a>

The set of relations in a database must be specified to the system by means of a data-definition language (DDL). The SQL DDL allows specification of not only a set of relations, but also information about each relation, including:

* The schema for each relation.
* The types of values associated with each attribute.
* The integrity constraints.
* The set of indices to be maintained for each relation.
* The security and authorization information for each relation.
* The physical storage structure of each relation on disk.

## <a name="Domain Types in SQL">Domain Types in SQL</a>

* **char(n)**. Fixed length character string, with user-specified length n.
* **varchar(n)**. Variable length character strings, with user-specified maximum length n.
* **int**. Integer (a finite subset of the integers that is machine-dependent).
* **smallint**. Small integer (a machine-dependent subset of the integer domain type).
* **numeric(p,d)**. A fixed-point number with user-specified precision. The number consists of p digits (plus a sign), and d of the p digits are to the right of the decimal point.
* **real, double precision**. Floating point and double-precision floating point numbers, with machine-dependent precision.
* **float(n)**. Floating point number, with user-specified precision of at least n digits.

## <a name="Create Table Construct">Create Table Construct</a>

We define an SQL relation by using the `create table` command:

```sql
create table r(A1 D1, A2 D2, ..., An Dn,
                (integrity-constraint1),
                ...,
                (integrity-constraintk));
```

* $$r$$ is the name of the relation.
* each $$A_i$$ is an attribute name in the schema of relation $$r$$.
* $$D_i$$ is the data type of values in the domain of attribute $$A_i$$.

Example:

```sql
create table instructor (
                ID char(5),
                name varchar(20) not null,
                dept_name varchar(20),
                salary numeric(8,2));
```

#### Integrity Constraints in Create Table

* **not null**. The `not null` constraint on an attribute specifies that the null value is not allowed for that attribute; in other words, the constraint excludes the null value from the domain of that attribute.
* **primary key ($$A_1, A_2,..., A_m$$)**. The `primary key` specification says that attributes $$A_1, A_2,..., A_m$$ form the primary key for the relation. The primary key attributes are required to be _nonnull_ and _unique_; that is, no tuple can have a null value for a primary key attribute, and no two tuples in the relation can be equal on all the primary key attributes.
* **foreign key ($$A_1, A_2,..., A_m$$) references s**. The `foreign key` specification says that the values of attributes ($$A_1, A_2,..., A_m$$) for any tuple in the relation must correspond to values of the primary key attributes of some tuple in relation s.

Example: Declare _ID_ as the primary key for _instructor_

```sql
create table instructor (
                ID char(5),
                name varchar(20) not null,
                dept_name varchar(20),
                salary numeric(8,2),
                primary key (ID),
                foreign key (dept_name) references department);
```

SQL prevents any update to the database that violates an integrity constraint. For example, if a newly inserted or modified tuple in a relation has null values for any primary-key attribute, or if the tuple has the same value on the primary-key attributes as does another tuple in the relation, SQL flags an error and prevents the update.

A newly created relation is empty initially. We can use the `insert` command to load data into the relation.

```sql
insert into instructor
                values (10211, 'Smith', 'Biology', 66000);
```

We can use the `delete` command to delete tuples from a relation.

```sql
delete from student;
```

## <a name="Drop and Alter Table Constructs">Drop and Alter Table Constructs</a>

To remove a relation from an SQL database, we use the `drop table` command. The `drop table` command deletes all information about the dropped relation from the database.

```sql
drop table r;
```

The command `drop table r;` is a more drastic action than `delete from r;`. The latter retains relation r, but deletes all tuples in r. The former deletes not only all tuples of r, but also the schema for r.

We use the `alter table` command to add attributes to an existing relation. All tuples in the relation are assigned null as the value for the new attribute.

```sql
alter table r add A D;
```

where r is the name of an existing relation, A is the name of the attribute to be added, and D is the type of the added attribute.

## <a name="Basic Structure of SQL Queries">Basic Structure of SQL Queries</a>

The basic structure of an SQL query consists of three clauses: `select`, `from`, and `where`. The query takes as its input the relations listed in the `from` clause, operates on them as specified in the `where` and `select` clauses, and then produces a relation as the result.

```sql
select A1, A2,..., An
from r1, r2,..., rm
where P;
```

where $$A_i$$ represents an attribute, $$R_i$$ represents a relation, and $$P$$ is a predicate. The result of an SQL query is a relation.

## <a name="The select Clause">The select Clause</a>

The `select` clause list the attributes desired in the result of a query. Note: SQL names are case insensitive

Example: find the names of all instructors:

```sql
select name
from instructor;
```

SQL allows duplicates in relations as well as in query results. To force the elimination of duplicates, insert the keyword `distinct` after `select`.

Example: Find the names of all departments with instructor, and remove duplicates.

```sql
select distinct dept_name
from instructor;
```

SQL also allows us to use the keyword `all` to specify explicitly that duplicates are not removed. But since duplicate retention is the default, we shall not use `all` in our examples. To ensure the elimination of duplicates in the results of our example queries, we shall use `distinct` whenever it is necessary.

```sql
select all dept_name
from instructor;
```

An **asterisk** in the select clause denotes “all attributes”

```sql
select *
from instructor;
```

The `select` clause may also contain **arithmetic expressions** involving the operators +, −, ∗, and / operating on constants or attributes of tuples.

Example: return a relation that is the same as the instructor relation, except that the value of the attribute salary is divided by 12.

```sql
select ID, name, salary/12
from instructor;
```

## <a name="The where Clause">The where Clause</a>

The `where` clause allows us to select only those rows in the result relation of the `from` clause that satisfy a specified predicate.

Example: Find the names of all instructors in the Computer Science department who have salary greater than $70,000

```sql
select name
from instructor
where dept_name = 'Comp. Sci.' and salary > 70000;
```

SQL allows the use of the logical connectives `and`, `or`, and `not` in the `where` clause. The operands of the logical connectives can be expressions involving the comparison operators <, <=, >, >=, =, and <>.

## <a name="Queries on Multiple Relations">Queries on Multiple Relations</a>

Queries often need to access information from multiple relations. An an example, suppose we want to answer the query “Retrieve the names of all instructors, along with their department names and department building name.”

In SQL, to answer the above query, we list the relations that need to be accessed in the `from` clause, and specify the matching condition in the `where` clause. The above query can be written in SQL as

```sql
select name, instructor.dept_name, building
from instructor, department
where instructor.dept_name = department.dept_name;
```

Note that the attribute *dept_name* occurs in both the relations *instructor* and *department*, and the relation name is used as a prefix (in *instructor.dept_name* and *department.dept_name*) to make clear to which attribute we are referring. In contrast, the attributes *name* and *building* appear in only one of the relations, and therefore do not need to be prefixed by the relation name.

A typical SQL query has the form:

```sql
select A1, A2,..., An
from r1, r2,..., rm
where P;
```

Although the clauses must be written in the order `select`, `from`, `where`, the easiest way to understand the operations specified by the query is to consider the clauses in operational order: first `from`, then `where`, and then `select`. In general, the meaning of an SQL query can be understood as follows:

1. Generate a Cartesian product of the relations listed in the `from` clause
2. Apply the predicates specified in the `where` clause on the result of Step 1.
3. For each tuple in the result of Step 2, output the attributes (or results of expressions) specified in the `select` clause.

The above sequence of steps helps make clear what the result of an SQL query should be, not how it should be executed.

## <a name="Natural Join">Natural Join</a>

**Natural join** considers only those pairs of tuples with the same value on those attributes that appear in the schemas of both relations, and retains only one copy of each common column. So, going back to the example of the relations *instructor* and *teaches*, computing *instructor* natural join *teaches* considers only those pairs of tuples where both the tuple from *instructor* and the tuple from *teaches* have the same value on the common attribute, *ID*.

```sql
select name, course_id
from instructor, teaches
where instructor.ID = teaches.ID;
```

This query can be written more concisely using the natural-join operation in SQL as:

```sql
select name, course_id
from instructor natural join teaches;
```

**Danger** in natural join: beware of unrelated attributes with same name which get equated incorrectly. For example, suppose we wish to answer the query “List the names of instructors along with the titles of courses that they teach.” The query can be written in SQL as follows:

```sql
select name, title
from instructor natural join teaches, course
where teaches.course_id = course.course_id;
```

The natural join of *instructor* and *teaches* is first computed, as we saw earlier, and a Cartesian product of this result with *course* is computed, from which the `where` clause extracts only those tuples where the course identifier from the join result matches the course identifier from the *course* relation. Note that *teaches.course_id* in the `where` clause refers to the *course_id* field of the natural join result, since this field in turn came from the *teaches* relation.

In contrast the following SQL query does *not* compute the same result:

```sql
select name, title
from instructor natural join teaches natural join course;
```

To see why, note that the natural join of *instructor* and *teaches* contains the attributes *(ID, name, dept_name, salary, course_id, sec_id)*, while the *course* relation contains the attributes *(course_id, title, dept_name, credits)*. As a result, the natural join of these two would require that the *dept_name* attribute values from the two inputs be the same, in addition to requiring that the *course_id* values be the same. This query would then omit all (instructor name, course title) pairs where the instructor teaches a course in a department other than the instructor’s own department. The previous query, on the other hand, correctly outputs such pairs.

To provide the benefit of natural join while avoiding the danger of equating attributes erroneously, SQL provides a form of the natural join construct that allows you to specify exactly which columns should be equated. This feature is illustrated by the following query:

```sql
select name, title
from (instructor natural join teaches) join course using (course_id);
```

The operation `join ... using` requires a list of attribute names to be specified. Both inputs must have attributes with the specified names.

## <a name="The Rename Operation">The Rename Operation</a>

The SQL allows renaming relations and attributes using the `as` clause. The `as` clause can appear in both the `select` and `from` clauses.

```sql
old-name as new-name
```

For example, if we want the attribute name `name` to be replaced with the name `instructor_name`, we can rewrite the preceding query as:

```sql
select name as instructor_name, course_id
from instructor, teaches
where instructor.ID = teaches.ID;
```

The `as` clause is particularly useful in renaming relations. One reason to rename a relation is to replace a long relation name with a shortened version that is more convenient to use elsewhere in the query. To illustrate, we rewrite the query “For all instructors in the university who have taught some course, find their names and the course ID of all courses they taught.”

```sql
select T.name, S.course_id
from instructor as T, teaches as S
where T.ID = S.ID;
```

Another reason to rename a relation is a case where we wish to compare tuples in the same relation. We then need to take the Cartesian product of a relation with itself and, without renaming, it becomes impossible to distinguish one tuple from the other. Suppose that we want to write the query “Find the names of all instructors whose salary is greater than at least one instructor in the Biology department.” We can write the SQL expression:

```sql
select distinct T.name
from instructor as T, instructor as S
where T.salary > S.salary and S.dept_name = 'Biology';
```

## <a name="String Operations">String Operations</a>

SQL specifies strings by enclosing them in **single quotes**. A single quote character that is part of a string can be specified by using two single quote characters.

SQL includes a string-matching operator for comparisons on character strings. The operator `like` uses patterns that are described using two special characters:

* percent (`%`). The `%` character matches any substring.
* underscore (`_`). The `_` character matches any character.

Patterns are case sensitive; that is, uppercase characters do not match lowercase characters, or vice versa. To illustrate pattern matching, we consider the following examples:

* ’Intro%’ matches any string beginning with “Intro”.
* ’%Comp%’ matches any string containing “Comp” as a substring, for example, ’Intro. to Computer Science’, and ’Computational Biology’.
* ’___’ matches any string of exactly three characters.
* ’___%’ matches any string of at least three characters.


SQL expresses patterns by using the `like` comparison operator. Consider the query “Find the names of all departments whose building name includes the substring ‘Watson’.” This query can be written as:

```sql
select dept_name
from department
where building like '%Watson%';
```

For patterns to include the special pattern characters (that is, % and   ), SQL allows the specification of an escape character. The escape character is used immediately before a special pattern character to indicate that the special pattern character is to be treated like a normal character. We define the escape character for a `like` comparison using the escape keyword. To illustrate, consider the following patterns, which use a backslash (\) as the escape character:

* like ’ab\%cd%’ escape ’\’ matches all strings beginning with “ab%cd”.
* like ’ab\\cd%’ escape ’\’ matches all strings beginning with “ab\cd”.

SQL allows us to search for mismatches instead of matches by using the `not like` comparison operator.

## <a name="Ordering the Display of Tuples">Ordering the Display of Tuples</a>

The `order by` clause causes the tuples in the result of a query to appear in sorted order. To list in alphabetic order all instructors in the Physics department, we write:

```sql
select name
from instructor
where dept_name = 'Physics'
order by name;
```

By default, the `order by` clause lists items in **ascending** order. To specify the sort order, we may specify `desc` for descending order or `asc` for ascending order. Furthermore, ordering can be performed on **multiple attributes**. Suppose that we wish to list the entire *instructor* relation in descending order of *salary*. If several instructors have the same salary, we order them in ascending order by name.

```sql
select *
from instructor
order by salary desc, name asc;
```

## <a name="Where Clause Predicates">Where Clause Predicates</a>

SQL includes a `between` comparison operator to simplify `where` clauses that specify that a value be less than or equal to some value and greater than or equal to some other value. If we wish to find the names of instructors with salary amounts between $90,000 and $100,000, we can use the between comparison to write:

```sql
select name
from instructor
where salary between 90000 and 100000;
```

SQL permits us to use the notation $$(v_1, v_2, ..., v_n)$$ to denote a tuple of arity n containing values $$v_1, v_2, ..., v_n$$. The comparison operators can be used on tuples, and the ordering is defined lexicographically. For example, $$(a_1, a_2) <= (b_1, b_2)$$ is true if $$a_1 <= b_1$$ **and** $$a_2 <= b_2$$; similarly, the two tuples are equal if all their attributes are equal.

```sql
select name, course_id
from instructor, teaches
where (instructor.ID, dept_name) = (teaches.ID, 'Biology');
```

## <a name="Set Operations">Set Operations</a>

The SQL operations `union`, `intersect`, and `except` operate on relations and correspond to the mathematical set-theory operations respectively.

To find the set of all courses taught either in Fall 2009 or in Spring 2010, or both

```sql
(select course_id from section where semester='Fall' and year=2009)
union
(select course_id from section where semester='Spring' and year=2010);
```

The `union` operation automatically eliminates duplicates, unlike the select clause. If we want to retain all duplicates, we must write `union all` in place of `union`.

To find the set of all courses taught in the Fall 2009 as well as in Spring 2010 we write:

```sql
(select course_id from section where semester='Fall' and year=2009)
intersect
(select course_id from section where semester='Spring' and year=2010);
```

The `intersect` operation automatically eliminates duplicates. If we want to retain all duplicates, we must write `intersect all` in place of `intersect`.

To find all courses taught in the Fall 2009 semester but not in the Spring 2010 semester, we write:

```sql
(select course_id from section where semester='Fall' and year=2009)
except
(select course_id from section where semester='Spring' and year=2010);
```

The `except` operation automatically eliminates duplicates in the inputs before performing set difference. If we want to retain duplicates, we must write `except all` in place of `except`.

## <a name="Null Values">Null Values</a>

Null values present special problems in relational operations, including arith- metic operations, comparison operations, and set operations. The result of an arithmetic expression (involving, for example +, −, ∗, or /) is null if any of the input values is null.

Since the predicate in a `where` clause can involve Boolean operations such as `and`, `or`, and `not` on the results of comparisons, the definitions of the Boolean operations are extended to deal with the value `unknown`.

* **and**: The result of *true* **and** *unknown* is *unknown*, *false* **and** *unknown* is *false*, while *unknown* **and** *unknown* is *unknown*.
* **or**: The result of *true* **or** *unknown* is *true*, *false* **or** *unknown* is unknown, while *unknown* **or** *unknown* is *unknown*.
* **not**: The result of **not** *unknown* is *unknown*.

If the `where` clause predicate evaluates to either `false` or `unknown` for a tuple, that tuple is not added to the result.

The predicate `is null` can be used to check for null values. An example is to find all instructors whose salary is null.

```sql
select name
from instructor
where salary is null;
```

## <a name="Aggregate Functions">Aggregate Functions</a>

*Aggregate functions* are functions that take a collection (a set or multiset) of values as input and return a single value. SQL offers five built-in aggregate functions:

* Average: `avg`
* Minimum: `min`
* Maximum: `max`
* Total: `sum`
* Count: `count`

The input to `sum` and `avg` must be a collection of numbers, but the other operators can operate on collections of nonnumeric data types, such as strings, as well.

“Find the average salary of instructors in the Computer Science department”:

```sql
select avg (salary) as avg_salary
from instructor
where dept_name='Comp.Sci.';
```

“Find the total number of instructors who teach a course in the Spring 2010 semester”:

```sql
select count (distinct ID)
from teaches
where semester='Spring' and year=2000;
```

"Find the number of tuples in the course relation":

```sql
select count (*)
from course;
```

#### Aggregation with Grouping

There are circumstances where we would like to apply the aggregate function not only to a single set of tuples, but also to a group of sets of tuples; we specify this wish in SQL using the `group by` clause. The attribute or attributes given in the `group by` clause are used to form groups. Tuples with the same value on all attributes in the `group by` clause are placed in one group.

"Find the average salary in each department”:

```sql
select dept_name, avg (salary) as avg_salary
from instructor
group by dept_name;
```

When an SQL query uses grouping, it is important to ensure that the only attributes that appear in the `select` statement without being aggregated are those that are present in the `group by` clause. In other words, any attribute that is not present in the `group by` clause must appear only inside an aggregate function if it appears in the `select` clause, otherwise the query is treated as erroneous. For example, the following query is erroneous since *ID* does not appear in the `group by` clause, and yet it appears in the `select` clause without being aggregated:

```sql
/* erroneous query */
select dept_name, ID, avg (salary)
from instructor
group by dept_name;
```

#### The Having Clause

At times, it is useful to state a condition that applies to groups rather than to tuples. To express such a query, we use the `having` clause of SQL. For example, we might be interested in only those departments where the average salary of the instructors is more than $42,000.

```sql
select dept_name, avg (salary) as avg_salary
from instructor
group by dept_name
having avg (salary) > 42000;
```

Predicates in the `having` clause are applied after the formation of groups whereas predicates in the `where` clause are applied before forming groups.

#### Aggregation with Null and Boolean Values

Null values, when they exist, complicate the processing of aggregate operators. For example, assume that some tuples in the *instructor* relation have a null value for *salary*. Consider the following query to total all salary amounts:

```sql
select sum(salary)
from instructor;
```

The values to be summed in the preceding query include null values, since some tuples have a null value for *salary*. Rather than say that the overall sum is itself *null*, the SQL standard says that the `sum` operator should ignore *null* values in its input.

In general, aggregate functions treat nulls according to the following rule: All aggregate functions except `count (*)` ignore null values in their input collection. As a result of null values being ignored, the collection of values may be empty. The `count` of an empty collection is defined to be 0, and all other aggregate operations return a value of null when applied on an empty collection. The effect of null values on some of the more complicated SQL constructs can be subtle.

A **Boolean** data type that can take values `true`, `false`, and `unknown`, was introduced in SQL:1999. The aggregate functions `some` and `every`, which mean exactly what you would intuitively expect, can be applied on a collection of Boolean values.

## <a name="Nested Subqueries">Nested Subqueries</a>

SQL provides a mechanism for the nesting of subqueries. A subquery is a **select-from-where** expression that is nested
within another query. A common use of subqueries is to perform tests for set membership, set comparisons, and set cardinality.

#### Set Membership

SQL allows testing tuples for membership in a relation. The `in` connective tests for set membership, where the set is a collection of values produced by a `select` clause. The `not in` connective tests for the absence of set membership.

As an illustration, reconsider the query “Find all the courses taught in the both the Fall 2009 and Spring 2010 semesters.” Earlier, we wrote such a query by intersecting two sets: the set of courses taught in Fall 2009 and the set of courses taught in Spring 2010. We can take the alternative approach of finding all courses that were taught in Fall 2009 and that are also members of the set of courses taught in Spring 2010. Clearly, this formulation generates the same results as the previous one did, but it leads us to write our query using the `in` connective of SQL. We begin by finding all courses taught in Spring 2010, and we write the subquery

```sql
(select course_id
 from section
 where semester='Spring' and year=2010)
```

We then need to find those courses that were taught in the Fall 2009 and that appear in the set of courses obtained in the subquery. We do so by nesting the subquery in the `where` clause of an outer query. The resulting query is

```sql
select distinct course_id
from section
where semester='Fall' and year=2009 and
    course_id in (select course_id
                  from section
                  where semester='Spring' and year=2010);
```

This example shows that it is possible to write the same query several ways in SQL. This flexibility is beneficial, since it allows a user to think about the query in the way that seems most natural.

We use the `not in` construct in a way similar to the `in` construct. For example, to find all the courses taught in the Fall 2009 semester but not in the Spring 2010 semester, we can write:

```sql
select distinct course_id
from section
where semester='Fall' and year=2009 and
    course_id not in (select course_id
                  from section
                  where semester='Spring' and year=2010);
```

The `in` and `not in` operators can also be used on enumerated sets. The following query selects the names of instructors whose names are neither “Mozart” nor “Einstein”.

```sql
select distinct name
from instructor
where name not in ('Mozart', 'Einstein')
```

#### Set Comparison

As an example of the ability of a nested subquery to compare sets, consider the previous query “Find the names of all instructors whose salary is greater than at least one instructor in the Biology department.” SQL does, however, offer an alternative style for writing the preceding query. The phrase “greater than at least one” is represented in SQL by `> some`. This construct allows us to rewrite the query in a form that resembles closely our formulation of the query in English.

```sql
select name
from instructor
where salary > some (select salary
                     from instructor
                     where dept_name='Biology');
```

The subquery generates the set of all salary values of all instructors in the Biology department. The `> some` comparison in the `where` clause of the outer `select` is true if the *salary* value of the tuple is greater than at least one member of the set of all salary values for instructors in Biology.

SQL also allows `< some`, `<= some`, `>= some`, `= some`, and `<>` some comparisons. And `= some` is identical to `in`, whereas `<> some` is *not* the same as `not in`.

Now we modify our query slightly. Let us find the names of all instructors that have a salary value greater than that of each instructor in the Biology department. The construct `> all` corresponds to the phrase “greater than all.” Using this construct, we write the query as follows:

```sql
select name
from instructor
where salary > all (select salary
                    from instructor
                    where dept_name='Biology');
```

As it does for `some`, SQL also allows `< all`, `<= all`, `>= all`, `= all`, and `<> all` comparisons.And `<> all` is identical to `not in`, whereas `= all` is *not* the same as `in`.

#### Test for Empty Relations

SQL includes a feature for testing whether a subquery has any tuples in its result. The `exists` construct returns the value **true** if the argument subquery is nonempty. Using the `exists` construct, we can write the query “Find all courses taught in both the Fall 2009 semester and in the Spring 2010 semester” in still another way:

```sql
select course_id
from section as S
where semester='Fall' and year=2009 and
    exists (select *
            from section as T
            where semester='Spring' and year=2010 and
                S.course_id=T.course_id);
```

The above query also illustrates a feature of SQL where a correlation name from an outer query (S in the above query), can be used in a subquery in the `where` clause. A subquery that uses a correlation name from an outer query is called a **correlated subquery**.

We can test for the nonexistence of tuples in a subquery by using the `not exists` construct.

#### Test for the Absence of Duplicate Tuples

SQL includes a boolean function for testing whether a subquery has duplicate tuples in its result. The `unique` construct returns the value **true** if the argument subquery contains no duplicate tuples. Using the `unique` construct, we can write the query “Find all courses that were offered at most once in 2009” as follows:

```sql
select T.course_id
from course as T
where unique (select R.course_id
              from section as R
              where T.course_id=R.course_id and
                    R.year=2009);
```

Note that if a course is not offered in 2009, the subquery would return an empty result, and the `unique` predicate would evaluate to true on the empty set.

#### Subqueries in the From Clause

SQL allows a subquery expression to be used in the `from` clause. The key concept applied here is that any `select-from-where` expression returns a relation as a result and, therefore, can be inserted into another `select-from-where` anywhere that a relation can appear.

Consider the query “Find the average instructors’ salaries of those departments where the average salary is greater than $42,000.” We wrote this query previously by using the `having` clause. We can now rewrite this query, without using the `having` clause, by using a subquery in the `from` clause, as follows:

```sql
select dept_name, avg_salary
from (select dept_name, avg (salary) as avg_salary
      from instructor
      group by dept_name)
where avg_salary > 42000;
```

Note that we do not need to use the `having` clause, since the subquery in the `from` clause computes the average salary, and the predicate that was in the `having` clause earlier is now in the `where` clause of the outer query.

We can give the subquery result relation a name, and rename the attributes, using the `as` clause, as illustrated below.

```sql
select dept_name, avg_salary
from (select dept_name, avg (salary)
      from instructor
      group by dept_name)
      as dept_avg (dept_name, agv_salary)
where avg_salary > 42000;
```

As another example, suppose we wish to find the maximum across all departments of the total salary at each department. The `having` clause does not help us in this task, but we can write this query easily by using a subquery in the `from` clause, as follows:

```sql
select max (tot_salary)
from (select dept_name, sum (salary)
      from instructor
      group by dept_name)
      as dept_total (dept_name, tot_salary);
```

#### The with Clause

The `with` clause provides a way of defining a temporary relation whose definition is available only to the query in which the `with` clause occurs. Consider the following query, which finds those departments with the maximum budget.

```sql
with max_budget (value) as
    (select max (budget)
     from department)
select budget
from department, max_budget
where department.budget=max_budget.value;
```

The `with` clause defines the temporary relation *max_budget*, which is used in the immediately following query.

#### Scalar Subqueries

SQL allows subqueries to occur wherever an expression returning a value is permitted, provided the subquery returns only one tuple containing a single attribute; such subqueries are called **scalar subqueries**. For example, a subquery can be used in the `select` clause as illustrated in the following example that lists all departments along with the number of instructors in each department:

```sql
select dept_name,
    (select count (*)
     from instructor
     where department.dept_name=instructor.dept_name)
     as num_instructors
from department;
```

The subquery in the above example is guaranteed to return only a single value since it has a `count(*)` aggregate without a `group by`.


## <a name="Modification of the Database">Modification of the Database</a>

We have restricted our attention until now to the extraction of information from the database. Now, we show how to add, remove, or change information with SQL.

#### Deletion

A delete request is expressed in much the same way as a query. We can delete only whole tuples; we cannot delete values on only particular attributes. SQL expresses a deletion by

```sql
delete from r
where P;
```

where P represents a predicate and r represents a relation. The `delete` statement first finds all tuples t in r for which P(t) is true, and then deletes them from r. The `where` clause can be omitted, in which case all tuples in r are deleted.

Note that a `delete` command operates on only one relation. If we want to delete tuples from several relations, we must use one `delete` command for each relation. The predicate in the `where` clause may be as complex as a `select` command’s `where` clause.

For example, "Delete all instructors with a salary between $13,000 and $15,000":

```sql
delete from instructor
where salary between 13000 and 15000;
```

#### Insertion

To insert data into a relation, we either specify a tuple to be inserted or write a query whose result is a set of tuples to be inserted. Obviously, the attribute values for inserted tuples must be members of the corresponding attribute’s domain. Similarly, tuples inserted must have the correct number of attributes.

The simplest `insert` statement is a request to insert one tuple.

```sql
insert into course
    values (’CS-437’, ’Database Systems’, ’Comp. Sci.’, 4);
```

In this example, the values are specified in the order in which the corresponding attributes are listed in the relation schema. For the benefit of users who may not remember the order of the attributes, SQL allows the attributes to be specified as part of the `insert` statement.

```sql
insert into course (title, course_id, credits, dept_name)
    values (’Database Systems’, ’CS-437’, 4, ’Comp. Sci.’);
```

More generally, we might want to insert tuples on the basis of the result of a query.

```sql
insert into instructor
    select ID, name, dept_name, 18000
    from student
    where dept_name = 'Music' and tot_cred > 144;
```

Instead of specifying a tuple as we did earlier in this section, we use a `select1 to specify a set of tuples. SQL evaluates the `select` statement first, giving a set of tuples that is then inserted into the *instructor* relation.

#### Update

In certain situations, we may wish to change a value in a tuple without changing all values in the tuple. For this purpose, the `update` statement can be used. As we could for `insert` and `delete`, we can choose the tuples to be updated by using a query.

Suppose that annual salary increases are being made, and salaries of all instructors are to be increased by 5 percent. We write:

```sql
update instructor
set salary = salary * 1.05;
```

The preceding update statement is applied once to each of the tuples in instructor relation.

If a salary increase is to be paid only to instructors with salary of less than $70,000, we can write:

```sql
update instructor
set salary = salary * 1.05
where salary < 70000;
```
