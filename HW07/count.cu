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
	thrust::device_vector<int> d_sort = d_in;
	thrust::sort(d_sort.begin(), d_sort.end());
	thrust::device_vector<int> count_frequency(d_sort.size());
	thrust::fill(count_frequency.begin(), count_frequency.end(), (int)1);

	auto new_size = thrust::reduce_by_key(thrust::device, d_sort.begin(), d_sort.begin() + d_sort.size(), count_frequency.begin(), values.begin(), counts.begin());
	values.resize(new_size.first - values.begin());
	counts.resize(new_size.second - counts.begin());
}
