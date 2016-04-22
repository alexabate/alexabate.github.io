---
layout: post
title: "Database plan"
tags:
    - python
    - notebook
--- 
# Database plan

Below is the current simple representation of the planned schema for the TucsonEventFinder
![png]({{ BASE_PATH }}/images/dbplan.png)

A couple of things require some extra explanation: the `hours_map` field in the `event_occurrences` table and the `service_flag` field in the `venues` table are planned to be a bitmask or bitmap represented as integers. For example the `service_flag` will refer to an array of 5 elements that indicate whether the venue serves beer, coffee, food, liquor, wine; the element will take 0 for no and 1 for yes. The actual number stored will be the decimal integer that corresponds to this binary representation. Similarly `hours_map` will refer to an array of 48 elements that represent every half hour period during a day, with 0 meaning the venue is closed and 1 meaning the venue is open. So with just 5 bits the integer stored will vary between 0 and 31, and with 48 bit it will vary between 0 and 281474976710655. An `hours_map` type field is also what is stored for each day of the week in the `hours` table.


The flow chart below gives an idea of how the database will be filled on a weekly basis

![png]({{ BASE_PATH }}/images/flow.png)
