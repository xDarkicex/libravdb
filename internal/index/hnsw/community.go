package hnsw

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// CommunityBounds contains the exact center and maximum radius of a community.
type CommunityBounds struct {
	Mean   []float32
	Radius float32
}

type CommunityRegistry struct {
	Bounds     map[uint32]*CommunityBounds
	NodeToComm []uint32
}

// ComputeCommunities performs a stateless iterative Louvain pass over the HNSW graph
// to cluster nodes into communities based on topological edges.
func (idx *Index) ComputeCommunities(ctx context.Context, budget int) (*CommunityRegistry, error) {
	// 1. Snapshot the graph size to avoid touching new nodes inserted concurrently
	maxNode := uint32(idx.nodes.Len())

	// If the graph is empty, return empty registry
	if maxNode == 0 {
		return &CommunityRegistry{
			Bounds:     make(map[uint32]*CommunityBounds),
			NodeToComm: make([]uint32, 0),
		}, nil
	}

	numEdges := 0.0
	for i := uint32(1); i < maxNode; i++ {
		node := idx.nodes.Get(i)
		if node != nil {
			numEdges += float64(len(idx.getNodeLinks(node, 0)))
		}
	}
	if numEdges == 0 {
		numEdges = 1.0
	}

	// Calculate tau = 1 / (2 * |E|)
	tau := 1.0 / (2.0 * numEdges)
	if tau == 0 || math.IsNaN(tau) {
		tau = 0.0001
	}

	// 2. Initialize communities: each node starts in its own community
	nodeToComm := make([]uint32, maxNode)
	commDeg := make(map[uint32]float64, maxNode)  // Sigma_tot
	nodeDeg := make(map[uint32]float64, maxNode)  // k_i
	
	// Pre-calculate degrees to save time
	for i := uint32(1); i < maxNode; i++ {
		node := idx.nodes.Get(i)
		if node == nil {
			continue
		}
		links := idx.getNodeLinks(node, 0)
		deg := float64(len(links))
		nodeToComm[i] = i
		commDeg[i] = deg
		nodeDeg[i] = deg
	}

	// 3. Iterative Louvain Passes
	moved := true
	passes := 0
	nodesTouched := 0

	for moved && passes < 10 && nodesTouched < budget {
		moved = false
		passes++

		for i := uint32(1); i < maxNode; i++ {
			if nodesTouched >= budget {
				break
			}
			nodesTouched++

			node := idx.nodes.Get(i)
			if node == nil {
				continue
			}

			currentComm := nodeToComm[i]
			ki := nodeDeg[i]
			if ki == 0 {
				continue
			}

			// Find neighboring communities and k_i,in
			links := idx.getNodeLinks(node, 0)
			neighborComms := make(map[uint32]float64) // comm -> weight
			for _, neighborID := range links {
				if neighborID < maxNode {
					nComm := nodeToComm[neighborID]
					neighborComms[nComm] += 1.0
				}
			}

			// Remove node from current community
			commDeg[currentComm] -= ki

			bestComm := currentComm
			bestDeltaQ := 0.0

			for nComm, kiIn := range neighborComms {
				sigmaTot := commDeg[nComm]
				dq := (kiIn / (2.0 * numEdges)) - ((sigmaTot * ki) / (4.0 * numEdges * numEdges))
				
				if dq > bestDeltaQ && dq > tau {
					bestDeltaQ = dq
					bestComm = nComm
				}
			}

			// Assign to best community
			nodeToComm[i] = bestComm
			commDeg[bestComm] += ki
			if bestComm != currentComm {
				moved = true
			}
		}
	}

	// 4. Compute bounding structures for the resulting communities
	// We'll collect members of each community and compact IDs to [1, C]
	commMembers := make(map[uint32][]uint32)
	remapped := make(map[uint32]uint32)
	nextCommID := uint32(1)
	
	for nodeID, oldCommID := range nodeToComm {
		if nodeID == 0 {
			continue // Sentinel
		}
		newCommID, ok := remapped[oldCommID]
		if !ok {
			newCommID = nextCommID
			remapped[oldCommID] = newCommID
			nextCommID++
		}
		nodeToComm[nodeID] = newCommID
		commMembers[newCommID] = append(commMembers[newCommID], uint32(nodeID))
	}

	bounds := make(map[uint32]*CommunityBounds)
	for commID, members := range commMembers {
		// Calculate mean
		mean := make([]float32, idx.config.Dimension)
		for _, memberID := range members {
			node := idx.nodes.Get(memberID)
			if node == nil {
				continue
			}
			vec, err := idx.getNodeVector(node)
			if err != nil || len(vec) != idx.config.Dimension {
				continue
			}
			for d := 0; d < idx.config.Dimension; d++ {
				mean[d] += vec[d]
			}
		}
		
		count := float32(len(members))
		if count == 0 {
			continue
		}
		for d := 0; d < idx.config.Dimension; d++ {
			mean[d] /= count
		}
		
		// Calculate max radius
		var maxRadius float32
		for _, memberID := range members {
			node := idx.nodes.Get(memberID)
			if node == nil {
				continue
			}
			vec, err := idx.getNodeVector(node)
			if err != nil || len(vec) != idx.config.Dimension {
				continue
			}
			dist := idx.distance(mean, vec)
			if dist > maxRadius {
				maxRadius = dist
			}
		}
		
		bounds[commID] = &CommunityBounds{
			Mean:   mean,
			Radius: maxRadius,
		}
	}

	return &CommunityRegistry{
		Bounds:     bounds,
		NodeToComm: nodeToComm,
	}, nil
}

// Serialize encodes the CommunityRegistry into a binary format.
func (c *CommunityRegistry) Serialize() ([]byte, error) {
	var buf bytes.Buffer
	
	// Magic bytes and version
	buf.Write([]byte("LIBRACMM"))
	if err := binary.Write(&buf, binary.LittleEndian, uint32(1)); err != nil {
		return nil, err
	}
	
	// Write lengths
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(c.Bounds))); err != nil {
		return nil, err
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(c.NodeToComm))); err != nil {
		return nil, err
	}
	
	// Write Bounds map
	for commID, bounds := range c.Bounds {
		if err := binary.Write(&buf, binary.LittleEndian, commID); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(bounds.Mean))); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, bounds.Mean); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, bounds.Radius); err != nil {
			return nil, err
		}
	}
	
	// Write NodeToComm
	if err := binary.Write(&buf, binary.LittleEndian, c.NodeToComm); err != nil {
		return nil, err
	}
	
	return buf.Bytes(), nil
}

// DeserializeCommunityRegistry decodes a CommunityRegistry from binary format.
func DeserializeCommunityRegistry(data []byte) (*CommunityRegistry, error) {
	buf := bytes.NewReader(data)
	
	magic := make([]byte, 8)
	if _, err := io.ReadFull(buf, magic); err != nil {
		return nil, fmt.Errorf("failed to read magic bytes: %w", err)
	}
	if string(magic) != "LIBRACMM" {
		return nil, fmt.Errorf("invalid magic bytes: expected LIBRACMM, got %s", magic)
	}
	
	var version uint32
	if err := binary.Read(buf, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported community registry version: %d", version)
	}
	
	var boundsLen, nodeToCommLen uint32
	if err := binary.Read(buf, binary.LittleEndian, &boundsLen); err != nil {
		return nil, err
	}
	if err := binary.Read(buf, binary.LittleEndian, &nodeToCommLen); err != nil {
		return nil, err
	}
	
	bounds := make(map[uint32]*CommunityBounds, boundsLen)
	for i := uint32(0); i < boundsLen; i++ {
		var commID uint32
		if err := binary.Read(buf, binary.LittleEndian, &commID); err != nil {
			return nil, err
		}
		
		var meanLen uint32
		if err := binary.Read(buf, binary.LittleEndian, &meanLen); err != nil {
			return nil, err
		}
		
		mean := make([]float32, meanLen)
		if err := binary.Read(buf, binary.LittleEndian, &mean); err != nil {
			return nil, err
		}
		
		var radius float32
		if err := binary.Read(buf, binary.LittleEndian, &radius); err != nil {
			return nil, err
		}
		
		bounds[commID] = &CommunityBounds{
			Mean:   mean,
			Radius: radius,
		}
	}
	
	nodeToComm := make([]uint32, nodeToCommLen)
	if err := binary.Read(buf, binary.LittleEndian, &nodeToComm); err != nil {
		return nil, err
	}
	
	return &CommunityRegistry{
		Bounds:     bounds,
		NodeToComm: nodeToComm,
	}, nil
}
