#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts){
	//copying input to another vector because it is const and cannot be sorted
	thrust::device_vector<int> d_sort = d_in;
	//sorting input data
	thrust::sort(d_sort.begin(), d_sort.end());
	//extra vector filled with all 1s to calculate the occurrences of each number
	thrust::device_vector<int> count_frequency(d_sort.size());
	thrust::fill(count_frequency.begin(), count_frequency.end(), (int)1);

	//calling reduce_by_key on input
	//values has values of all distinct entries in input
	///counts has #occurrences of each unique entry
	auto new_size = thrust::reduce_by_key(thrust::device, d_sort.begin(), d_sort.begin() + d_sort.size(), count_frequency.begin(), values.begin(), counts.begin());
	//resizing output
	values.resize(new_size.first - values.begin());
	counts.resize(new_size.second - counts.begin());
}
