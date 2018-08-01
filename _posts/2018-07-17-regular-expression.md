---
layout:         post
title:          Regular Expressions
subtitle:
card-image:     /mldl/assets/images/cards/cat4.gif
date:           2018-07-17 09:00:00
tags:           [linux]
categories:     [linux]
post-card-type: image
mathjax:        true
---

* <a href="#Anchors">Anchors</a>
* <a href="#Brackets">Brackets</a>
* <a href="#Quantifiers">Quantifiers</a>
* <a href="#Character classes">Character classes</a>
* <a href="#Special character">Special character</a>
* <a href="#Pattern modifier">Pattern modifier</a>
* <a href="#Grouping and capturing">Grouping and capturing</a>
* <a href="#Greedy and Lazy match">Greedy and Lazy match</a>
* <a href="#Boundaries">Boundaries</a>
* <a href="#Back-references">Back-references</a>
* <a href="#Look-ahead and look-behind">Look-ahead and look-behind</a>

## <a name="Anchors">Anchors `^` `$`</a>

* `^The` matches any string that starts with "The"
* `end$` matches any string that ends with "end"
* `^The end$` exact string match (starts and ends with "The end")
* `roar` matches any string that has the text "roar" in it

## <a name="Brackets">Brackets `[]`</a>

* `[abc]` find any character ("a" or "b" or "c") between the brackets
* `[a-c]` same as previous
* `[^abc]` find any character **NOT** between the brackets
* `[0-9]%` find any digit between 0 to 9 before a "%" sign
* `[^0-9]` find any non-digit between the brackets
* `[a-fA-F0-9]` find any character that represents a single hexadecimal digit, case insensitively
* `[^a-zA-Z]` find any character that is not a letter from "a" to "z" or from "A" to "Z"
* `(x|y)` find any character that is either "x" or "y"

Remember that inside bracket expressions all special characters (including the backslash `\`) lose their special powers: thus we will not apply the “escape rule”.

## <a name="Quantifiers">Quantifiers `*` `+` `?` `{}`</a>

* `abc*` matches a string that has "ab" followed by zero or more "c"
* `abc+` matches a string that has "ab" followed by one or more "c"
* `abc?` matches a string that has "ab" followed by zero or one "c"
* `abc{2}` matches a string that has "ab" followed by 2 "c"
* `abc{2,}` matches a string that has "ab" followed by 2 or more "c"
* `abc{2,5}` matches a string that has "ab" followed by 2 up to 5 "c"
* `a(bc)*` matches a string that has "a" followed by zero or more copies of the sequence "bc"
* `a(bc){2,5}` matches a string that has "a" followed by 2 up to 5 copies of the sequence "bc"

## <a name="Character classes">Character classes `\d` `\w` `s`</a>

* `\d` matches a single character that is a digit
* `\w` matches a word (alphanumeric characters plus underscore)
* `\s` matches a whitespace character (includes tabs and line breaks)

`\d`, `\w` and `\s` also present their negations with `\D`, `\W` and `\S` respectively. For example, `\D` will perform the inverse match with respect to that obtained with `\d`.

* `\D` matches a single non-digit character
* `\W` matches a non-word
* `\S` matches a non-whitespace
* `.` matches any character

In order to be taken literally, you must escape the characters `^` `.` `[` `]` `$` `(` `)` `|` `*` `+` `?` `{` `\` with a backslash `\` as they have special meaning.


## <a name="Special character">Special character</a>

* `\n` new line
* `\t` tab
* `\r` carriage return

## <a name="Pattern modifier">Pattern modifier</a>

* `i` perform case-insensitive matching
* `g` perform a global match (find all matches rather than stopping after the first match)
* `m` perform multiline matching

## <a name="Grouping and capturing">Grouping and capturing `()`</a>

* `a(bc)` parentheses create a capturing group with value "bc"
* `a(?:bc)*` using `?:` we disable the capturing group
* `a(?<foo>bc)` using `?<foo>` we put a name "foo" to the group

## <a name="Greedy and Lazy match">Greedy and Lazy match</a>

The quantifiers `*` `+` `{}` are greedy operators, so they expand the match as far as they can through the provided text.  For example, `<.+>` matches "<div>simple div</div>" in "This is a <div> simple div</div> test". In order to catch only the "div" tag we can use a `?` to make it lazy:

* `<.+?>` matches any character one or more times, in a lazy way
* `<.*?>` matches any character zero or more times, in a lazy away

## <a name="Boundaries">Boundaries `\b` `\B`</a>

* `\bis\b` performs a "whole words only" search. Valid match: "This island is beautiful."

`\b` represents an anchor like caret matching positions where one side is a word character (like `\w`) and the other side is not a word character.

It comes with the its negation, `\B`. This matches all positions where `\b` doesn’t match and could be if we want to find a search pattern fully surrounded by word characters.

* `\Bla\B` matches only if the pattern is fully surrounded by word characters. Valid match: "island".

## <a name="Back-references">Back-references</a>

* `([abc])\1` using `\1` matches the same text that was matched by the first capturing group. For example, "cc" and "aa" in "abccbaacacb".
* `([abc])([de])\2\1` we can use `\2` (`\3`, `\4`, etc.) to identify the same text that was matched by the second (third, fourth, etc.) capturing group. For example, "adda" and "ceec" in "adda abceec".
* `(?<foo>[abc])\k<foo>` we put the name "foo" to the group and we reference it later (`\k<foo>`). The result is the same of the first regex. For example, "cc" and "aa" in "abccbaacacb".

## <a name="Look-ahead and look-behind">Look-ahead and look-behind</a>

* `d(?=r)` matches a "d" only if it is followed by "r", but "r" will not be part of the overall regex match
* `(?<=r)d` matches a "d" only if it is preceded by an "r", but "r" will not be part of the overall regex match

You can use also the negation operator `!`

* `d(?!r)` matches a "d" only if it is not followed by "r", but "r" will not be part of the overall regex match
* `(?<!r)d` matches a "d" only if is not preceded by an "r", but "r" will not be part of the overall regex match

References:

* [<u>Regex tutorial — A quick cheatsheet by examples</u>](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)
* [<u>Regular Expression Cheat Sheet</u>](https://github.com/niklongstone/regular-expression-cheat-sheet)
