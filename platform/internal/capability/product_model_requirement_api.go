package capability

// DescriptorRequiredModels returns the exact learned ModelRelease requirements
// declared by one stable product capability. Non-product capabilities return no
// product-model requirements.
func DescriptorRequiredModels(descriptor Descriptor) ([]ProductModelRequirement, error) {
	return descriptorRequiredModels(descriptor)
}
