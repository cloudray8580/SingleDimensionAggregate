#pragma once
#ifndef HILBERT_hpp
#define HILBERT_hpp

/* define the bitmask_t type as an integer of sufficient size */
typedef unsigned long long bitmask_t;
/* define the halfmask_t type as an integer of 1/2 the size of bitmask_t */
typedef unsigned long halfmask_t;

#define adjust_rotation(rotation,nDims,bits)                            \
do {                                                                    \
      /* rotation = (rotation + 1 + ffs(bits)) % nDims; */              \
      bits &= -bits & nd1Ones;                                          \
      while (bits)                                                      \
        bits >>= 1, ++rotation;                                         \
      if ( ++rotation >= nDims )                                        \
        rotation -= nDims;                                              \
} while (0)

#define ones(T,k) ((((T)2) << (k-1)) - 1)

#define rdbit(w,k) (((w) >> (k)) & 1)

#define rotateRight(arg, nRots, nDims)                                  \
((((arg) >> (nRots)) | ((arg) << ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define rotateLeft(arg, nRots, nDims)                                   \
((((arg) << (nRots)) | ((arg) >> ((nDims)-(nRots)))) & ones(bitmask_t,nDims))

#define DLOGB_BIT_TRANSPOSE
static bitmask_t
bitTranspose(unsigned nDims, unsigned nBits, bitmask_t inCoords)
#if defined(DLOGB_BIT_TRANSPOSE)
{
	unsigned const nDims1 = nDims - 1;
	unsigned inB = nBits;
	unsigned utB;
	bitmask_t inFieldEnds = 1;
	bitmask_t inMask = ones(bitmask_t, inB);
	bitmask_t coords = 0;

	while ((utB = inB / 2))
	{
		unsigned const shiftAmt = nDims1 * utB;
		bitmask_t const utFieldEnds =
			inFieldEnds | (inFieldEnds << (shiftAmt + utB));
		bitmask_t const utMask =
			(utFieldEnds << utB) - utFieldEnds;
		bitmask_t utCoords = 0;
		unsigned d;
		if (inB & 1)
		{
			bitmask_t const inFieldStarts = inFieldEnds << (inB - 1);
			unsigned oddShift = 2 * shiftAmt;
			for (d = 0; d < nDims; ++d)
			{
				bitmask_t in = inCoords & inMask;
				inCoords >>= inB;
				coords |= (in & inFieldStarts) << oddShift++;
				in &= ~inFieldStarts;
				in = (in | (in << shiftAmt)) & utMask;
				utCoords |= in << (d*utB);
			}
		}
		else
		{
			for (d = 0; d < nDims; ++d)
			{
				bitmask_t in = inCoords & inMask;
				inCoords >>= inB;
				in = (in | (in << shiftAmt)) & utMask;
				utCoords |= in << (d*utB);
			}
		}
		inCoords = utCoords;
		inB = utB;
		inFieldEnds = utFieldEnds;
		inMask = utMask;
	}
	coords |= inCoords;
	return coords;
}
#else
{
	bitmask_t coords = 0;
	unsigned d;
	for (d = 0; d < nDims; ++d)
	{
		unsigned b;
		bitmask_t in = inCoords & ones(bitmask_t, nBits);
		bitmask_t out = 0;
		inCoords >>= nBits;
		for (b = nBits; b--;)
		{
			out <<= nDims;
			out |= rdbit(in, b);
		}
		coords |= out << d;
	}
	return coords;
}
#endif

bitmask_t hilbert_c2i(unsigned nDims, unsigned nBits, bitmask_t const coord[])
{
	if (nDims > 1)
	{
		unsigned const nDimsBits = nDims * nBits;
		bitmask_t index;
		unsigned d;
		bitmask_t coords = 0;
		for (d = nDims; d--; )
		{
			coords <<= nBits;
			coords |= coord[d];
		}

		if (nBits > 1)
		{
			halfmask_t const ndOnes = ones(halfmask_t, nDims);
			halfmask_t const nd1Ones = ndOnes >> 1; /* for adjust_rotation */
			unsigned b = nDimsBits;
			unsigned rotation = 0;
			halfmask_t flipBit = 0;
			bitmask_t const nthbits = ones(bitmask_t, nDimsBits) / ndOnes;
			coords = bitTranspose(nDims, nBits, coords);
			coords ^= coords >> nDims;
			index = 0;
			do
			{
				halfmask_t bits = (coords >> (b -= nDims)) & ndOnes;
				bits = rotateRight(flipBit ^ bits, rotation, nDims);
				index <<= nDims;
				index |= bits;
				flipBit = (halfmask_t)1 << rotation;
				adjust_rotation(rotation, nDims, bits);
			} while (b);
			index ^= nthbits >> 1;
		}
		else
			index = coords;
		for (d = 1; d < nDimsBits; d *= 2)
			index ^= index >> d;
		return index;
	}
	else
		return coord[0];
}

#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

typedef enum { up_ = 0, left_, right_, down_, none_dir } direction;
typedef enum { lowerleft = 0, lowerright, upperright, upperleft, none_quad } quadrant;
typedef enum { clockwise = 0, counterclockwise, unavailable } clockdirection;

// first level is the original direction(i.e., up, left, right, down)
// the second level is the counter clock wise quadrant, start from lowerleft
direction next_level_direction[4][4] = { { up_, up_, right_, left_ },{ down_, left_, left_, up_ },{ right_, down_, up_, right_ },{ left_, right_, down_, down_ } };

// first level is the original direction(i.e., up, left, right, down)
// the second level is the sub region direction(i.e., up, left, right, down)
clockdirection clock_direction[4][4] = { { clockwise, counterclockwise, counterclockwise, unavailable },{ clockwise, counterclockwise, unavailable, clockwise },{ clockwise, unavailable, counterclockwise, clockwise },{ unavailable, counterclockwise, counterclockwise, clockwise } };

// define a type name interval and define a struct named interval
typedef struct interval {
	int lower_x;
	int upper_x;
	int lower_y;
	int upper_y;
	bitmask_t lower_hilbert_value;
	bitmask_t upper_hilbert_value;
}interval;

//const int NumDims = 2;
//const int NumBits = 4;

// 2 dimensions
// the max_bits here is the bits used for the Hilbert
// all input should be int
// down_x, up_x, down_y, up_y denotes the rectangle region, includes itself
// center lies in the upper right corner
void inline sub_range_decompose(unsigned max_bits, unsigned current_bits, unsigned using_bits, int down_x, int up_x, int down_y, int up_y, int center_x, int center_y, direction pre, direction dir, quadrant quad_, vector<interval> &intervals) { // here we should use stack instead

	// debug info
	//if (intervals.size() == 635) {
	//	cout << "debug here !" << endl;
	//}
	//cout << down_x << " " << up_x << " " << down_y << " " << up_y << endl;
	//cout << up_x - down_x << " " << up_y - down_y << endl;
	//if ((up_x - down_x > (1 << current_bits)) || (up_y - down_y > (1 << current_bits))) {
	//	cout << "debug here 2!" << endl;
	//}

	if (current_bits == 0) {
		interval inter;
		inter.lower_x = down_x;
		inter.upper_x = up_x;
		inter.lower_y = down_y;
		inter.upper_y = up_y;
		bitmask_t coord[2];
		coord[0] = inter.lower_x;
		coord[1] = inter.lower_y;
		inter.lower_hilbert_value = hilbert_c2i(2, max_bits, coord);
		inter.upper_hilbert_value = hilbert_c2i(2, max_bits, coord);
		intervals.push_back(inter);
		return;
	}

	bool square_region = false;

	// if meet the using_bits constraint
	if (current_bits == max_bits - using_bits) {
		square_region = true;
	}

	// if it covers the entire region or one quarter of the region
	double half_region = 1 << (current_bits - 1);
	if (down_x == center_x - half_region && down_y == center_y - half_region && up_x == center_x + half_region - 1 && up_y == center_y + half_region - 1) {
		square_region = true;
	}
	/*else {
		switch (quad_) {
		case lowerleft:
			if (down_x == center_x-half_region && up_x == center_x-1 && down_y == center_y-half_region && up_y == center_y-1) {
				square_region = true;
			}
			break;
		case lowerright:
			if (down_x == center_x && up_x == center_x+half_region-1 && down_y == center_y-half_region && up_y == center_y-1) {
				square_region = true;
			}
			break;
		case upperright:
			if (down_x == cente r_x && up_x == center_x+half_region-1 && down_y == center_y && up_y == center_y+half_region-1) {
				square_region = true;
			}
			break;
		case upperleft:
			if (down_x == center_x-half_region && up_x == center_x-1 && down_y == center_y && up_y == center_y+half_region-1 ) {
				square_region = true;
			}
			break;
		}
	}*/

	clockdirection clockdir = clock_direction[pre][dir];
	if (current_bits == max_bits) {
		clockdir = counterclockwise;
	}

	if (square_region) {
		// return this inverval
		interval inter;
		bitmask_t coord[2];
		switch (clockdir) {
		case clockwise:
			switch (dir) {
			case up_:
				inter.lower_x = up_x;
				inter.upper_x = down_x;
				inter.lower_y = up_y;
				inter.upper_y = up_y;
				break;
			case left_:
				inter.lower_x = down_x;
				inter.upper_x = down_x;
				inter.lower_y = up_y;
				inter.upper_y = down_y;
				break;
			case right_:
				inter.lower_x = up_x;
				inter.upper_x = up_x;
				inter.lower_y = down_y;
				inter.upper_y = up_y;
				break;
			case down_:
				inter.lower_x = down_x;
				inter.upper_x = up_x;
				inter.lower_y = down_y;
				inter.upper_y = down_y;
				break;
			}
			break;
		case counterclockwise:
			switch (dir) {
			case up_:
				inter.lower_x = down_x;
				inter.upper_x = up_x;
				inter.lower_y = up_y;
				inter.upper_y = up_y;
				break;
			case left_:
				inter.lower_x = down_x;
				inter.upper_x = down_x;
				inter.lower_y = down_y;
				inter.upper_y = up_y;
				break;
			case right_:
				inter.lower_x = up_x;
				inter.upper_x = up_x;
				inter.lower_y = up_y;
				inter.upper_y = down_y;
				break;
			case down_:
				inter.lower_x = up_x;
				inter.upper_x = down_x;
				inter.lower_y = down_y;
				inter.upper_y = down_y;
				break;
			}
			break;
		case unavailable:
			// do noting
			break;
		}
		coord[0] = inter.lower_x;
		coord[1] = inter.lower_y;
		inter.lower_hilbert_value = hilbert_c2i(2, max_bits, coord);
		coord[0] = inter.upper_x;
		coord[1] = inter.upper_y;
		inter.upper_hilbert_value = hilbert_c2i(2, max_bits, coord);

		intervals.push_back(inter);
		return;
	}
	else {
		current_bits--;
		half_region = 1 << (current_bits - 1);
		if (current_bits == 0) { // the half_region will be a negative number in such case
			half_region = 0.1; // using 0.1 to let (+) be the same while (-) minus 1
		}
		direction parent_dir;
		direction child_dir;
		quadrant quad;

		// return the intervals in subregions under hilbert order
		switch (clockdir) {
		case clockwise:
			switch (dir) {
			case up_:
				parent_dir = up_;
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case left_:
				parent_dir = left_;
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case right_:
				parent_dir = right_;
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case down_:
				parent_dir = down_;
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			}
			break;
		case counterclockwise:
			switch (dir) {
			case up_:
				parent_dir = up_;
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case left_:
				parent_dir = left_;
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case right_:
				parent_dir = right_;
				if (up_x >= center_x && up_y >= center_y) { // upper right
					quad = upperright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = upperleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), max(center_y, down_y), up_y, center_x - half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			case down_:
				parent_dir = down_;
				if (up_x >= center_x && down_y < center_y) { // lower right
					quad = lowerright;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, down_y, min(center_y - 1, up_y), center_x + half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (up_x >= center_x && up_y >= center_y) { // upper right
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, max(center_x, down_x), up_x, max(center_y, down_y), up_y, center_x + half_region, center_y + half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && up_y >= center_y) { // upper left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				if (down_x < center_x && down_y < center_y) { // lower left
					quad = lowerleft;
					child_dir = next_level_direction[parent_dir][quad];
					sub_range_decompose(max_bits, current_bits, using_bits, down_x, min(center_x - 1, up_x), down_y, min(center_y - 1, up_y), center_x - half_region, center_y - half_region, parent_dir, child_dir, quad, intervals);
				}
				break;
			}
			break;
		case unavailable:
			// do noting
			break;
		}
	}
}

void inline mergeContinousIntervals(vector<interval> &intervals) {

	vector<interval> result;
	bitmask_t temp_upper;
	int i = 0, j = 0;
	while (i < intervals.size()) {
		temp_upper = intervals[i].upper_hilbert_value;
		j = 1;
		while (i + j < intervals.size() && intervals[i + j].lower_hilbert_value == temp_upper + 1) {
			temp_upper = intervals[i + j].upper_hilbert_value;
			j++;
		}
		interval inter;
		inter.lower_x = intervals[i].lower_x;
		inter.lower_y = intervals[i].lower_y;
		inter.lower_hilbert_value = intervals[i].lower_hilbert_value;
		inter.upper_x = intervals[i + j - 1].upper_x;
		inter.upper_y = intervals[i + j - 1].upper_y;
		inter.upper_hilbert_value = intervals[i + j - 1].upper_hilbert_value;
		result.push_back(inter);
		i += j;
	}
	intervals = result;
}

void inline GetIntervals(unsigned max_bits, unsigned using_bits, unsigned lower_x, unsigned upper_x, unsigned lower_y, unsigned upper_y, vector<interval> &intervals) {
	intervals.clear();
	sub_range_decompose(max_bits, max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, 1 << (max_bits - 1), 1 << (max_bits - 1), none_dir, left_, none_quad, intervals);
	mergeContinousIntervals(intervals);
}

int inline getTotalIntervalsCount(unsigned max_bits, unsigned using_bits, unsigned lower_x, unsigned upper_x, unsigned lower_y, unsigned upper_y) {
	vector<interval> intervals;
	sub_range_decompose(max_bits, max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, 1 << (max_bits - 1), 1 << (max_bits - 1), none_dir, left_, none_quad, intervals);
	mergeContinousIntervals(intervals);
	return intervals.size();
}

// based on first level left hilbert cureve
//int main(int argc, const char * args[]) {
//
//	bitmask_t coord[2] = { 2244891,2941279 };
//	cout << hilbert_c2i(2, 22, coord) << endl;
//	//vector<interval> intervals;
//	//sub_range_decompose(4, 4, 2, 10, 3, 13, 8, 8, none_dir, left_, none_quad, intervals); // remember to change the 2 const when adjust the value
//	//cout << "before merge continous intervals: total count " << intervals.size() << endl;
//	//for (int i = 0; i < intervals.size(); i++) {
//	//	cout << "interval_" << i << ":	lower_x: " << intervals[i].lower_x << "	lower_y: " << intervals[i].lower_y << "	lower_hilbert: " << intervals[i].lower_hilbert_value << "	upper_x: " << intervals[i].upper_x << "	upper_y: " << intervals[i].upper_y << "	upper_hilbert: " << intervals[i].upper_hilbert_value << endl;
//	//}
//
//	//intervals = mergeContinousIntervals(intervals);
//	//cout << "after merge continous intervals: total count " << intervals.size() << endl;
//
//	//for (int i = 0; i < intervals.size(); i++) {
//	//	cout << "interval_" << i << ":	lower_x: " << intervals[i].lower_x << "	lower_y: " << intervals[i].lower_y << "	lower_hilbert: " << intervals[i].lower_hilbert_value << "	upper_x: " << intervals[i].upper_x << "	upper_y: " << intervals[i].upper_y << "	upper_hilbert: " << intervals[i].upper_hilbert_value << endl;
//	//}
//
//	//cout << getTotalIntervalsCount(4, 2, 10, 3, 13) << endl;
//
//	system("pause");
//	return 0;
//}

#endif // HILBERT_hpp
